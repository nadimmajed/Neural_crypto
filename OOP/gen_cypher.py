# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:08:55 2017

@author: nadim
"""
class gen_cypher():
    def __init__(self,message,key):
        self.message = message
        self.table = key
        self.bit_message = []
    
    def s_to_bitlist(self,s):
        ords=(ord(c) for c in s)
        shifts =(7,6,5,4,3,2,1,0)
        return [(o >> shift) & 1 for o in ords for shift in shifts]

    def generate_cypher(self):
        cypher = []
        
        for i in range(0,16):
            cypher.append(self.s_to_bitlist(self.message)[i]^self.table[i])
        return cypher
    
    def generate_bit(self):
        cy= []
        for i in range(0,16):
            cy.append(self.s_to_bitlist(self.message)[i])   
        return cy
    
    def test(self):
        test_c = []
        for i in range(0,16):
            test_c.append(self.generate_cypher()[i]^self.table[i])

        return test_c

    def bitlist_to_chars(self,t):
        bi = iter(t)
        bytes = zip(*(bi,)*8)
        shifts = (7, 6, 5, 4, 3, 2, 1, 0)
        for byte in bytes:
            yield chr(sum(bit << s for bit, s in zip(byte, shifts)))

    def bitlist_to_s(self):
        return ''.join(self.bitlist_to_chars(self.test()))
