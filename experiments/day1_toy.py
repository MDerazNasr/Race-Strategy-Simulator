import simpy
import random

env = simpy.Environment()
FUEL_EFFECT = 0.02
WEAR_EFFECT_COMPOUND = 0.05
temp_adjustment = 0.005
standings = {}

def car(env, num, fuel_kg, tire_age, base_lap):
    total_time = 0
    for i in range(10):
        yield env.timeout(base_lap + fuel_kg*FUEL_EFFECT + tire_age*WEAR_EFFECT_COMPOUND + temp_adjustment + random.uniform(-1, 1))
        print(f"Car {num} finishes lap {i+1} at time {env.now}")
        fuel_kg -= 3
        tire_age += 1
    standings[num] = env.now


car1 = env.process(car(env, 1, 20, 10, 90))
car2 = env.process(car(env, 2, 12, 10, 90))
car3 = env.process(car(env, 3, 11, 10, 90))
car4 = env.process(car(env, 4, 13, 10, 90))
car5 = env.process(car(env, 5, 9, 10, 90))
env.run()
for i,n in enumerate(standings):
    print(f" {i+1}. Car {n} : {standings[n]}")
