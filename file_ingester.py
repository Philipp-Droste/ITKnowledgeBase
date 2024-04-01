import streamlit as st
import csv

file_path = 'data.csv'

with open(file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        dat1_value = row['jobpost']
        print(dat1_value)