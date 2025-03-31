#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:16:38 2025

@author: sgajre
"""

import os
import google.generativeai as genai
os.environ['GEMINI_KEY'] = 'AIzaSyC89SIO4EAO5sM0bHAALlHwWo0z6tCyfCU'

genai.configure(api_key=os.environ['GEMINI_KEY'] )

model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Explain how AI work")

print(response.text)