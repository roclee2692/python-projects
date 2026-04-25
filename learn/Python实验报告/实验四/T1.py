class Car:
	"""汽车类：包含颜色和百公里油耗。"""

	def __init__(self, color: str, csfe: float):
		self.color = color
		self.csfe = csfe

	def introduce(self):
		"""对象实例的自我介绍。"""
		print(f"你好，我是一辆汽车，颜色是{self.color}，百公里油耗是{self.csfe}L。")


if __name__ == "__main__":
	my_car = Car("白色", 7.2)
	my_car.introduce()
