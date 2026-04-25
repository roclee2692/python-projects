// auto-input.js
// 单窗口稳定流程：固定入口页 -> 手动登录一次并保存登录态 -> 进入目标页 -> 找编辑器 -> 逐字输入

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const CONFIG = {
  // ====== 按需修改区域（主要改这里）======
  entryUrl: 'https://pintia.cn',
  targetUrl: 'https://pintia.cn/problem-sets/2033365657884266496/exam/problems/type/7?problemSetProblemId=2033366689529155584',

  // 已完成首次登录保存后，保持 false 以复用登录态
  captureLoginState: false,
  storageStatePath: path.join(__dirname, 'playwright-auth.json'),
  manualPauseBeforeContinueMs: 120000, // 无需终端输入：预留手动登录和跳转时间（毫秒）

  // 如果目标页 URL 不稳定，就留空 targetUrl，然后在 navigateStepsSelectors 中配置点击路径
  navigateStepsSelectors: [
    // 示例：'a:has-text("题库")',
    // 示例：'a:has-text("目标题目")',
  ],

  // 编辑器候选选择器（按稳定性排序）
  inputSelectors: [
    'div.cm-content[contenteditable="true"]',
    '[contenteditable="true"][role="textbox"]',
    '.cm-editor .cm-content[contenteditable="true"]',
    'div.cm-content[role="textbox"]',
    'div.cm-content',
  ],

  // 推荐方式1：把多行代码放到文件（如 input_code.txt / main.cpp），这里填文件名
  textFilePath: 'input_code.txt',
  // 方式2：直接写在模板字符串里（支持多行）
  text: String.raw``,
  buttonSelector: '',
  pressEnterAfterInput: false,
  pauseBeforeInputMs: 30000, // 进入目标页后，留给你手动选择题目的时间
  waitAfterInputForManualSubmitMs: 60000, // 输入完成后，留给你手动提交和评测的时间
  // =======================================

  // 稳定性参数
  headless: false,
  slowMo: 40,
  actionTimeoutMs: 60000,
  pageLoadTimeoutMs: 45000,
  navigationRetryCount: 3,
  navigationRetryDelayMs: 1200,
  typeDelayMs: 80,
  afterInputDelayMs: 400,
  afterSubmitDelayMs: 1200,
  closeBrowserWhenDone: false,
  keepBrowserOpenOnError: true,
};

function normalizeSelector(selector) {
  if (!selector) return '';
  const s = String(selector).trim();
  if (s === '{{BUTTON_SELECTOR}}') return '';
  return s;
}

function getInputText() {
  const maybePath = String(CONFIG.textFilePath || '').trim();
  if (maybePath) {
    const resolved = path.isAbsolute(maybePath)
      ? maybePath
      : path.join(__dirname, maybePath);
    if (!fs.existsSync(resolved)) {
      throw new Error(`未找到输入文件: ${resolved}`);
    }
    // 统一为 \n，避免 Windows CRLF 导致换行行为不一致
    return fs.readFileSync(resolved, 'utf8').replace(/\r\n/g, '\n');
  }

  return String(CONFIG.text || '').replace(/\r\n/g, '\n');
}

async function typeTextWithNewlines(page, editor, text, delayMs) {
  for (const ch of text) {
    if (ch === '\r') {
      continue;
    }
    if (ch === '\n') {
      await editor.press('Enter');
      if (delayMs > 0) {
        await page.waitForTimeout(delayMs);
      }
      continue;
    }
    await page.keyboard.type(ch, { delay: delayMs });
  }
}

async function maybeWaitNetworkIdle(page, ms = 10000) {
  try {
    await page.waitForLoadState('networkidle', { timeout: ms });
  } catch {
    console.log('[提示] networkidle 超时，继续执行（某些站点会持续请求）。');
  }
}

function isRetryableNavigationError(error) {
  const msg = String((error && error.message) || '').toLowerCase();
  return msg.includes('err_aborted') || msg.includes('frame was detached') || msg.includes('target page, context or browser has been closed');
}

async function safeGoto(context, pageRef, url, waitUntil, timeoutMs) {
  let page = pageRef;
  let lastError;

  for (let i = 1; i <= CONFIG.navigationRetryCount; i++) {
    try {
      if (!page || page.isClosed()) {
        page = await context.newPage();
        page.setDefaultTimeout(CONFIG.actionTimeoutMs);
        page.setDefaultNavigationTimeout(CONFIG.pageLoadTimeoutMs);
      }

      await page.goto(url, { waitUntil, timeout: timeoutMs });
      return page;
    } catch (error) {
      lastError = error;
      if (!isRetryableNavigationError(error) || i === CONFIG.navigationRetryCount) {
        throw error;
      }

      console.log(`[导航重试] 第 ${i}/${CONFIG.navigationRetryCount} 次失败，将重试: ${error.message}`);
      try {
        if (page && !page.isClosed()) {
          await page.close();
        }
      } catch {
        // 忽略关闭异常
      }
      page = await context.newPage();
      page.setDefaultTimeout(CONFIG.actionTimeoutMs);
      page.setDefaultNavigationTimeout(CONFIG.pageLoadTimeoutMs);
      await page.waitForTimeout(CONFIG.navigationRetryDelayMs);
    }
  }

  throw lastError;
}

async function findInputAcrossFrames(page, selectors, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let frameCount = 0;

  while (Date.now() < deadline) {
    const frames = page.frames();
    frameCount = frames.length;

    for (const frame of frames) {
      for (const selector of selectors) {
        const locator = frame.locator(selector).first();
        try {
          if (await locator.isVisible({ timeout: 250 })) {
            return { frame, selector, locator };
          }
        } catch {
          // 轮询模式下忽略短超时
        }
      }
    }

    await page.waitForTimeout(400);
  }

  throw new Error(`未在 ${timeoutMs}ms 内找到输入编辑器（已检查 ${frameCount} 个 frame）`);
}

async function goToTargetPage(context, page) {
  if (CONFIG.targetUrl) {
    console.log(`[导航] 进入目标页: ${CONFIG.targetUrl}`);
    page = await safeGoto(context, page, CONFIG.targetUrl, 'domcontentloaded', CONFIG.pageLoadTimeoutMs);
    await maybeWaitNetworkIdle(page);
    return page;
  }

  if (!CONFIG.navigateStepsSelectors.length) {
    console.log('[导航] 未配置 targetUrl 和点击路径，保留当前页继续。');
    return page;
  }

  console.log('[导航] 使用点击路径进入目标页面。');
  for (const selector of CONFIG.navigateStepsSelectors) {
    const step = page.locator(selector).first();
    await step.waitFor({ state: 'visible', timeout: CONFIG.actionTimeoutMs });
    await step.click();
    await page.waitForTimeout(500);
  }
  await maybeWaitNetworkIdle(page);
  return page;
}

async function createContext(browser) {
  const hasSavedState = fs.existsSync(CONFIG.storageStatePath);

  if (!CONFIG.captureLoginState && hasSavedState) {
    console.log(`[登录态] 复用: ${CONFIG.storageStatePath}`);
    return browser.newContext({ storageState: CONFIG.storageStatePath });
  }

  if (!CONFIG.captureLoginState && !hasSavedState) {
    console.log('[登录态] 未找到已保存状态，将进入手动登录流程。');
  }

  return browser.newContext();
}

async function ensureLoginState(context, page) {
  if (!CONFIG.captureLoginState) {
    return;
  }

  console.log(`[登录态] 请在当前窗口手动登录并进入目标页面，脚本将等待 ${Math.floor(CONFIG.manualPauseBeforeContinueMs / 1000)} 秒后自动继续。`);
  await page.waitForTimeout(CONFIG.manualPauseBeforeContinueMs);

  const dir = path.dirname(CONFIG.storageStatePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  await context.storageState({ path: CONFIG.storageStatePath });
  console.log(`[登录态] 已保存到: ${CONFIG.storageStatePath}`);
  console.log(`[登录态] 当前页面: ${page.url()}`);
}

async function inputText(page) {
  const selectors = [...new Set(CONFIG.inputSelectors.filter(Boolean))];
  if (!selectors.length) {
    throw new Error('inputSelectors 不能为空。');
  }

  console.log(`[定位] 候选编辑器选择器: ${selectors.join(' | ')}`);
  const matched = await findInputAcrossFrames(page, selectors, CONFIG.actionTimeoutMs);
  const editor = matched.locator;

  console.log(`[定位] 命中选择器: ${matched.selector}`);
  console.log(`[定位] 命中 frame: ${matched.frame.url() || 'about:blank'}`);

  await editor.scrollIntoViewIfNeeded();
  await editor.click({ timeout: CONFIG.actionTimeoutMs });

  const isEditable = await editor.evaluate((el) => el.isContentEditable);
  await page.keyboard.press('Control+A');
  await page.keyboard.press('Backspace');

  console.log('[输入] 逐字输入中...');
  if (isEditable) {
    await editor.pressSequentially(CONFIG.text, { delay: CONFIG.typeDelayMs });
  } else {
    await page.keyboard.type(CONFIG.text, { delay: CONFIG.typeDelayMs });
  }
}

async function submitIfNeeded(page, editor) {
  const buttonSelector = normalizeSelector(CONFIG.buttonSelector);

  if (buttonSelector) {
    console.log(`[提交] 点击按钮: ${buttonSelector}`);
    const btn = page.locator(buttonSelector).first();
    await btn.waitFor({ state: 'visible', timeout: CONFIG.actionTimeoutMs });
    await btn.click();
    return;
  }

  if (CONFIG.pressEnterAfterInput) {
    console.log('[提交] 按 Enter。');
    await editor.press('Enter');
    return;
  }

  console.log('[提交] 未配置提交动作，仅输入不提交。');
}

async function main() {
  let browser;
  let context;
  let page;
  let hasError = false;

  try {
    browser = await chromium.launch({
      headless: CONFIG.headless,
      slowMo: CONFIG.slowMo,
    });

    context = await createContext(browser);
    page = await context.newPage();
    page.setDefaultTimeout(CONFIG.actionTimeoutMs);
    page.setDefaultNavigationTimeout(CONFIG.pageLoadTimeoutMs);

    console.log(`[1/6] 打开入口页: ${CONFIG.entryUrl}`);
    page = await safeGoto(context, page, CONFIG.entryUrl, 'domcontentloaded', CONFIG.pageLoadTimeoutMs);
    await maybeWaitNetworkIdle(page);

    await ensureLoginState(context, page);

    console.log('[2/6] 进入目标页面');
    page = await goToTargetPage(context, page);

    if (CONFIG.pauseBeforeInputMs > 0) {
      console.log(`[2.5/6] 等待 ${Math.floor(CONFIG.pauseBeforeInputMs / 1000)} 秒，让你手动选择题目...`);
      await page.waitForTimeout(CONFIG.pauseBeforeInputMs);
    }

    console.log('[3/6] 定位编辑器并输入');
    const selectors = [...new Set(CONFIG.inputSelectors.filter(Boolean))];
    const matched = await findInputAcrossFrames(page, selectors, CONFIG.actionTimeoutMs);
    const editor = matched.locator;
    console.log(`[定位] 命中选择器: ${matched.selector}`);
    console.log(`[定位] 命中 frame: ${matched.frame.url() || 'about:blank'}`);

    const finalText = getInputText();
    if (!finalText) {
      throw new Error('输入内容为空。请设置 CONFIG.textFilePath 或 CONFIG.text。');
    }

    await editor.scrollIntoViewIfNeeded();
    await editor.click({ timeout: CONFIG.actionTimeoutMs });
    const isEditable = await editor.evaluate((el) => el.isContentEditable);
    await page.keyboard.press('Control+A');
    await page.keyboard.press('Backspace');
    console.log(`[输入] 内容长度: ${finalText.length} 字符`);

    if (isEditable) {
      await typeTextWithNewlines(page, editor, finalText, CONFIG.typeDelayMs);
    } else {
      await page.keyboard.type(finalText, { delay: CONFIG.typeDelayMs });
    }

    await page.waitForTimeout(CONFIG.afterInputDelayMs);

    console.log('[4/6] 执行提交动作（可选）');
    await submitIfNeeded(page, editor);

    if (CONFIG.waitAfterInputForManualSubmitMs > 0) {
      console.log(`[4.5/6] 等待 ${Math.floor(CONFIG.waitAfterInputForManualSubmitMs / 1000)} 秒，供你手动提交和查看评测...`);
      await page.waitForTimeout(CONFIG.waitAfterInputForManualSubmitMs);
    }

    await page.waitForTimeout(CONFIG.afterSubmitDelayMs);
    console.log('[5/6] 当前 URL: ' + page.url());
    console.log('[6/6] 执行完成');
  } catch (error) {
    hasError = true;
    console.error('\n[错误] 自动输入失败:');
    console.error(error && error.stack ? error.stack : error);

    try {
      if (page && !page.isClosed()) {
        await page.screenshot({ path: 'playwright_error.png', fullPage: true });
        console.log('[提示] 已保存错误截图: playwright_error.png');
      }
    } catch {
      // 截图失败可忽略
    }

    process.exitCode = 1;
  } finally {
    if (browser && CONFIG.closeBrowserWhenDone && !(hasError && CONFIG.keepBrowserOpenOnError)) {
      await browser.close();
    }
  }
}

main();