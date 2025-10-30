import simpy
import random

env = simpy.Environment()
FUEL_EFFECT = 0.02
WEAR_EFFECT_COMPOUND = 0.05
temp_adjustment = 0.005
standings = {}
# tire_compound = { #type of tire: [lap time adjustment, wear rate]
#     'soft': [-2, 2],
#     'medium': [-1, 1],
#     'hard': [0, 0.5]
# }
lap_adjustment = -1
wear_rate = 1

def car(env, num, fuel_kg, tire_age, base_lap):
    original_fuel_kg = fuel_kg
    current_tire_type = 'MEDIUM'  # Start with medium tires

    for i in range(10):
        lap_adjustment = get_lap_adjustment(current_tire_type)
        yield env.timeout(base_lap + fuel_kg*FUEL_EFFECT + tire_age*WEAR_EFFECT_COMPOUND + temp_adjustment + random.uniform(-1, 1) + lap_adjustment)
        print(f"Car {num} finishes lap {i+1} at time {env.now}")
        fuel_kg -= 1
        tire_age += tire_wear(current_tire_type)
        if tire_age >= 5 or fuel_kg == 0:
            yield from pitstop(env, num)
            tire_age = 0
            fuel_kg = original_fuel_kg
    standings[num] = env.now

def pitstop(env, num):
    pitstop_time = random.randint(1,5)
    print(f"Car {num} stops for a pitstop at time {env.now}")
    tire_type = get_tire_compound()
    print(f"Car {num} now has {tire_type} tires")
    yield env.timeout(pitstop_time)
    print(f"Car {num} leaves pitstop at time {env.now}")
    return tire_type  # Return the new tire type

def get_tire_compound():
    tire_type = random.choice(['SOFT', 'MEDIUM', 'HARD'])
    return tire_type

def get_lap_adjustment(tire_type):
    if tire_type == 'SOFT':
        return -2
    elif tire_type == 'MEDIUM':
        return -1
    else:  # HARD
        return 0

def tire_wear(tire_type):
    if tire_type == 'SOFT':
        return 2
    elif tire_type == 'MEDIUM':
        return 1
    else:  # HARD
        return 0.5

car1 = env.process(car(env, 1, 9, 0, 90))
car2 = env.process(car(env, 2, 9, 0, 90))
car3 = env.process(car(env, 3, 9, 0, 90))
car4 = env.process(car(env, 4, 9, 0, 90))
car5 = env.process(car(env, 5, 9, 0, 90))
env.run()
for i,n in enumerate(standings):
    print(f" {i+1}. Car {n} : {standings[n]}")

