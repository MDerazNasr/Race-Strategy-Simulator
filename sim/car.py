class Car:
    #tire_compound = {SOFT: ..., MEDIUM: ..., HARD: ...}
    
    def __init__(self, carid, base_lap_time, tire_compound, tire_age, fuel_load, fuel_starting_load, lap_times, total_time, pitstops):
        self.carid = carid
        self.base_lap_time = base_lap_time
        self.tire_compound = tire_compound
        self.tire_age = tire_age
        self.fuel_load = fuel_load
        self.fuel_starting_load = fuel_starting_load
        self.lap_times = lap_times
        self.total_time = total_time
        self.pitstops = pitstops

    def get_carid(self):
        return self.carid