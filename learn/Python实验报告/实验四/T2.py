class Car:
	"""汽车父类：颜色 + 百公里油耗。"""

	def __init__(self, color: str, csfe: float):
		self.color = color
		self.csfe = csfe


class Racing_Car(Car):
	"""赛车类：新增百公里加速秒数和最高时速。"""

	def __init__(self, color: str, csfe: float, accel_0_100: float, max_speed: int):
		super().__init__(color, csfe)
		self.accel_0_100 = accel_0_100
		self.max_speed = max_speed


class Bus(Car):
	"""公交车类：新增载人数。"""

	def __init__(self, color: str, csfe: float, passenger_capacity: int):
		super().__init__(color, csfe)
		self.passenger_capacity = passenger_capacity


if __name__ == "__main__":
	racing_car = Racing_Car("红色", 12.8, 3.1, 320)
	bus = Bus("蓝白色", 24.6, 56)

	print("Racing_Car 对象创建成功：")
	print(
		f"颜色: {racing_car.color}, 百公里油耗: {racing_car.csfe}L, "
		f"百公里加速: {racing_car.accel_0_100}s, 最高时速: {racing_car.max_speed}km/h"
	)

	print("Bus 对象创建成功：")
	print(
		f"颜色: {bus.color}, 百公里油耗: {bus.csfe}L, "
		f"载人数: {bus.passenger_capacity}人"
	)
