from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
import sys


def get_exif_data(image_path):
    """获取图片的 EXIF 数据，兼容无 EXIF 的图片。"""
    exif_data = {}
    with Image.open(image_path) as img:
        exif_info = img.getexif()
        if exif_info:
            for tag_id, value in dict(exif_info).items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    gps_info = {}
                    # 某些图片中 GPSInfo 是偏移值，需通过 GPS IFD 读取
                    if hasattr(value, "items"):
                        gps_items = value.items()
                    else:
                        gps_ifd = exif_info.get_ifd(0x8825) if hasattr(exif_info, "get_ifd") else {}
                        gps_items = gps_ifd.items()

                    for gps_tag, gps_value in gps_items:
                        sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                        gps_info[sub_tag] = gps_value
                    exif_data[tag] = gps_info
                else:
                    exif_data[tag] = value
    return exif_data

def convert_gps_to_degrees(gps_coords):
    """将 GPS 坐标转换为十进制度数，兼容 tuple/IFDRational。"""

    def to_float(x):
        # 兼容 (num, den) 与 IFDRational
        if isinstance(x, tuple):
            return x[0] / x[1]
        return float(x)

    degrees = to_float(gps_coords[0])
    minutes = to_float(gps_coords[1]) / 60.0
    seconds = to_float(gps_coords[2]) / 3600.0
    return degrees + minutes + seconds


def get_location_from_gps(gps_info):
    """根据 GPS 信息获取经纬度。"""
    if not gps_info:
        return None
    required_keys = ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]
    if not all(k in gps_info for k in required_keys):
        return None

    lat = convert_gps_to_degrees(gps_info["GPSLatitude"])
    lat_ref = gps_info["GPSLatitudeRef"]
    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode("utf-8", errors="ignore")
    if lat_ref == "S":
        lat = -lat

    lon = convert_gps_to_degrees(gps_info["GPSLongitude"])
    lon_ref = gps_info["GPSLongitudeRef"]
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode("utf-8", errors="ignore")
    if lon_ref == "W":
        lon = -lon
    return (lat, lon)


# 主程序
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    input_arg = sys.argv[1] if len(sys.argv) > 1 else "land"
    input_path = Path(input_arg)

    # 规则：绝对路径直接使用；相对路径优先按“脚本目录”解析
    if input_path.is_absolute():
        image_path = input_path
    else:
        # 无后缀时，自动尝试常见图片后缀
        if input_path.suffix:
            image_path = script_dir / input_path
        else:
            candidates = [
                script_dir / f"{input_path.name}.jpg",
                script_dir / f"{input_path.name}.JPG",
                script_dir / f"{input_path.name}.jpeg",
                script_dir / f"{input_path.name}.JPEG",
                script_dir / f"{input_path.name}.png",
                script_dir / f"{input_path.name}.PNG",
            ]
            image_path = next((p for p in candidates if p.exists()), candidates[0])

    try:
        exif_data = get_exif_data(image_path)

        if not exif_data:
            print("该图片没有 EXIF 元数据。")
            print("常见原因：")
            print("1. 图片来自社交软件/截图，EXIF 已被清除")
            print("2. 图片被二次编辑后重新导出")
            print("3. 拍摄设备未开启定位")
            print("建议：使用手机原图（未压缩）再试。")
        else:
            print(f"EXIF 字段数量: {len(exif_data)}")

            # 打印拍摄时间
            date_time = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
            if date_time:
                print(f"拍摄时间: {date_time}")
            else:
                print("未找到拍摄时间信息")

            # 打印 GPS 坐标
            gps_info = exif_data.get("GPSInfo")
            if gps_info:
                location = get_location_from_gps(gps_info)
                if location:
                    print(f"GPS坐标: 纬度 {location[0]:.6f}, 经度 {location[1]:.6f}")
                else:
                    print("找到 GPSInfo，但字段不完整，无法解析经纬度")
            else:
                print("未找到 GPS 信息（该图片可能未开启定位拍摄）")
    except FileNotFoundError:
        print(f"文件不存在: {image_path}")
        print("用法示例:")
        print("1) python painting.py")
        print("2) python painting.py land.jpg")
        print("3) python painting.py land")
        print("4) python painting.py " + str(script_dir / "land.JPG"))
    except Exception as e:
        print(f"读取图片失败: {e}")
