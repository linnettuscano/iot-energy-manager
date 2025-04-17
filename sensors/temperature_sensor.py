import random

def read_temperature_humidity():
    
    temperature = round(random.uniform(18, 30), 2)
    humidity = round(random.uniform(30, 70), 2)
    return {"temperature": temperature, "humidity": humidity}
