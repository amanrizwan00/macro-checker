#!/usr/bin/env python3
"""
Script: Extract USDA JSON Data to SQLite Database
Purpose: Load the USDA JSON file and extract nutrition data into a clean database

Usage: python extract_json_to_database.py
Input: FoodData_Central_foundation_food_json_*.json
Output: nutrition_database.db
"""

import json
import sqlite3
import os

def load_usda_json_file(filename):
    """
    Load the USDA JSON file that contains the FoundationFoods array
    """
    print(f"Loading JSON file: {filename}")
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # USDA structure: {"FoundationFoods": [ ... ]}
        foods_data = data.get("FoundationFoods", [])
        print(f"Successfully loaded {len(foods_data)} food items")
        return foods_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []

def create_nutrition_database(foods_data):
    """
    Create SQLite database and insert food nutrition data
    """
    print("Creating nutrition database...")
    
    conn = sqlite3.connect('nutrition_database.db')
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS foods')
    cursor.execute('''
        CREATE TABLE foods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fdc_id INTEGER,
            food_name TEXT,
            calories_per_100g REAL,
            protein_per_100g REAL,
            fat_per_100g REAL,
            carbs_per_100g REAL
        )
    ''')
    
    successful_inserts = 0
    
    for food_item in foods_data:
        try:
            food_name = food_item.get('description', '').strip()
            fdc_id = food_item.get('fdcId', 0)
            if not food_name:
                continue
            
            calories = protein = fat = carbs = 0.0
            
            for nutrient in food_item.get('foodNutrients', []):
                nutrient_info = nutrient.get('nutrient', {})
                nutrient_name = nutrient_info.get('name', '').lower()
                unit_name = nutrient_info.get('unitName', '').lower()
                amount = nutrient.get('amount', 0) or 0
                
                if 'energy' in nutrient_name and unit_name == 'kcal':
                    calories = amount
                elif nutrient_name == 'protein':
                    protein = amount
                elif 'total lipid' in nutrient_name:
                    fat = amount
                elif 'carbohydrate, by difference' in nutrient_name:
                    carbs = amount
            
            if calories > 0 or protein > 0 or fat > 0 or carbs > 0:
                cursor.execute('''
                    INSERT INTO foods (fdc_id, food_name, calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (fdc_id, food_name.lower(), calories, protein, fat, carbs))
                successful_inserts += 1
                
        except Exception as e:
            print(f"Error processing {food_item.get('description', 'Unknown')}: {e}")
    
    conn.commit()
    
    cursor.execute('SELECT COUNT(*) FROM foods')
    total_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT food_name, calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g FROM foods LIMIT 5')
    samples = cursor.fetchall()
    
    conn.close()
    
    print(f"\nDatabase created successfully!")
    print(f"Total foods inserted: {successful_inserts}/{len(foods_data)}")
    print(f"Database file: nutrition_database.db")
    
    print("\nSample foods:")
    for food_name, calories, protein, fat, carbs in samples:
        print(f"- {food_name}")
        print(f"  Calories: {calories} kcal/100g | Protein: {protein} g | Fat: {fat} g | Carbs: {carbs} g")
        print()
    
    return successful_inserts

def main():
    print("="*60)
    print("USDA JSON to Database Extractor")
    print("="*60)
    
    json_filename = '../Raw-data/FoodData_Central_foundation_food_json_2025-04-24.json'
    foods_data = load_usda_json_file(json_filename)
    
    if not foods_data:
        print("No data loaded. Please check your JSON file.")
        return
    
    count = create_nutrition_database(foods_data)
    
    if count > 0:
        print(f"\n✅ Success! {count} foods extracted to nutrition_database.db")
    else:
        print("\n❌ Failed to extract any foods. Please check JSON parsing.")

if __name__ == "__main__":
    main()
