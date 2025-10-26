import simpy
import random

env = simpy.Environment()
FUEL_EFFECT = 0.02
WEAR_EFFECT_COMPOUND = 0.05
temp_adjustment = 0.005
tire_type = 'SOFT'
standings = {}
tire_compound = { #type of tire: [lap time adjustment, wear rate]
    'soft': [-2, 2],
    'medium': [-1, 1],
    'hard': [0, 0.5]
}
lap_adjustment = -2
wear_rate = 2

def car(env, num, fuel_kg, tire_age, base_lap):
    original_fuel_kg = fuel_kg
    for i in range(10):
        yield env.timeout(base_lap + fuel_kg*FUEL_EFFECT + tire_age*WEAR_EFFECT_COMPOUND + temp_adjustment + random.uniform(-1, 1))
        print(f"Car {num} finishes lap {i+1} at time {env.now}")
        fuel_kg -= 1
        tire_age += tire_wear(env, num)
        if tire_age >= 5 or fuel_kg == 0:
            yield from pitstop(env, num)
            tire_age = 0
            fuel_kg = original_fuel_kg
    standings[num] = env.now

def pitstop(env, num):
    pitstop_time = random.randint(1,5)
    print(f"Car {num} stops for a pitstop at time {env.now}")
    tire_type = get_tire_compound(env, num)
    print(f"Car {num} now has {tire_type} tires")
    yield env.timeout(pitstop_time)
    print(f"Car {num} leaves pitstop at time {env.now}")

def get_tire_compound(env, num):
    tire_type = random.choice(['SOFT', 'MEDIUM', 'HARD'])
    return tire_type

def tire_wear(env, num):
    if tire_type == 'SOFT':
        return 2
    elif tire_type == 'MEDIUM':
        return 1
    elif tire_type == 'HARD':
        return 0.5
# def tire_wear(env, num):
#     tire_type = random.choice(['SOFT', 'MEDIUM', 'HARD'])
#     '''
#     How should tire compounds affect lap time?
#     Soft: -2 seconds per lap but wears 2x faster?
#     Medium: -1 second per lap, normal wear?
#     Hard: +0 seconds per lap but wears 0.5x slower
#     '''
#     if tire_type == 'SOFT':
#         lap_adjustment = -2
#         wear_rate = 2
#     elif tire_type == 'MEDIUM':
#         lap_adjustment = -1
#         wear_rate = 1
#     elif tire_type == 'HARD':
#         lap_adjustment = 0
#         wear_rate = 0.5
#     return lap_adjustment, wear_rate



car1 = env.process(car(env, 1, 20, 0, 90))
car2 = env.process(car(env, 2, 12, 0, 90))
car3 = env.process(car(env, 3, 11, 0, 90))
car4 = env.process(car(env, 4, 13, 0, 90))
car5 = env.process(car(env, 5, 9, 0, 90))
env.run()
for i,n in enumerate(standings):
    print(f" {i+1}. Car {n} : {standings[n]}")

