import random

class Car:

    #Class level constants
    FUEL_EFFECT = 0.02
    WEAR_EFFECT = 0.05
    #Tire compound data
    TIRE_DATA = {
        'SOFT': { 'modifier': -2, 'wear_rate': 2 },
        'MEDIUM': { 'modifier': -1, 'wear_rate': 1 },
        'HARD': { 'modifier': 0, 'wear_rate': 0.5 }
    }
    
    def __init__(self, carid, base_lap_time, tire_compound='MEDIUM', fuel_starting_load=100):
        #Basic attributes
        self.carid = carid
        self.base_lap_time = base_lap_time
        self.tire_compound = tire_compound
        self.fuel_starting_load = fuel_starting_load
        
        #Current state variables
        self.tire_age = 0
        self.fuel_load = fuel_starting_load
        
        #statistics
        self.lap_times = []
        self.total_time = 0
        self.pitstops = 0
        self.total_laps = 0
        self.total_pitstops = 0
        self.total_fuel_used = 0
        self.total_wear = 0
        self.total_temp_adjustment = 0
        self.total_random_adjustment = 0
        self.total_lap_adjustment = 0
        self.total_wear_rate = 0
        self.total_fuel_effect = 0
        self.total_wear_effect = 0
        self.total_temp_adjustment = 0  

        #physics parameters
        self.rho = 1.225 #density of air
        self.Cd = 0.3 #drag coefficient (0.7?)
        self.A = 2 #frontal area (1.5)
        self.g = 9.81 #acceleration due to gravity
        self.m = 1000 #mass of the car
        self.I = 1000 #moment of inertia
        self.r = 0.3 #radius of the tire
        self.f = 0.01 #rolling resistance coefficient
        self.Crr = 0.01 #rolling resistance coefficient
        self.P = 12000 #power of the engine (kW)
        self.mu = 1.5 #coefficient of friction
        self.Cl = 3.5 #downforce coefficient



    def get_carid(self):
        return self.carid