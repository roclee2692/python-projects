class Car:
	"""汽车类：包含颜色和百公里油耗。"""

	def __init__(self, color: str, csfe: float):
		self.color = color
		self.csfe = csfe

	def introduce(self):
		print(f"我是普通汽车，颜色{self.color}，百公里油耗{self.csfe}L。")


class Racing_Car(Car):
	"""赛车类：重写自我介绍。"""

	def __init__(self, color: str, csfe: float, accel_0_100: float, max_speed: int):
		super().__init__(color, csfe)
		self.accel_0_100 = accel_0_100
		self.max_speed = max_speed

	def introduce(self):
		print(
			f"我是赛车，颜色{self.color}，百公里油耗{self.csfe}L，"
			f"百公里加速{self.accel_0_100}s，最高时速{self.max_speed}km/h。"
		)


class Bus(Car):
	"""公交车类：重写自我介绍。"""

	def __init__(self, color: str, csfe: float, passenger_capacity: int):
		super().__init__(color, csfe)
		self.passenger_capacity = passenger_capacity

	def introduce(self):
		print(
			f"我是公交车，颜色{self.color}，百公里油耗{self.csfe}L，"
			f"可载客{self.passenger_capacity}人。"
		)


if __name__ == "__main__":
	racing = Racing_Car("黄色", 13.0, 3.0, 340)
	bus = Bus("绿色", 24.0, 60)

	racing.introduce()
	bus.introduce()
