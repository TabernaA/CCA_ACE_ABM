'''
model/classes/vintage.py

A non-agent class for machine vintages (capital goods)

'''
import random
class Vintage():
    def __init__(self, prod, amount = 1):
        self.productivity = prod
        self.amount = amount
        self.age = 0 
        self.lifetime = 19 #15 + random.randint(1,10)  # from lsd "eta" = 19