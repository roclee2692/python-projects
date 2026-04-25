class Car:
	"""汽车类：包含颜色和百公里油耗。"""

	def __init__(self, color: str, csfe: float):
		self.color = color
		self.csfe = csfe

	def introduce(self):
		print(f"我是普通汽车，颜色{self.color}，百公里油耗{self.csfe}L。")


class Racing_Car(Car):
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
	def __init__(self, color: str, csfe: float, passenger_capacity: int):
		super().__init__(color, csfe)
		self.passenger_capacity = passenger_capacity

	def introduce(self):
		print(
			f"我是公交车，颜色{self.color}，百公里油耗{self.csfe}L，"
			f"可载客{self.passenger_capacity}人。"
		)


def repeat_introduce(car_obj: Car, times: int):
	"""多态函数：接收父类或任一子类实例，重复调用自我介绍。"""
	if times <= 0:
		print("次数必须是正整数。")
		return

	for i in range(times):
		print(f"第{i + 1}次介绍：")
		car_obj.introduce()


if __name__ == "__main__":
	normal_car = Car("银色", 8.1)
	racing = Racing_Car("红色", 12.3, 3.1, 335)
	bus = Bus("白蓝", 26.0, 58)

	repeat_introduce(normal_car, 1)
	repeat_introduce(racing, 2)
	repeat_introduce(bus, 2)
