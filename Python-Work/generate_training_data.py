#!/usr/bin/env python3
"""
Script 2: Generate Training Data from Database
Purpose: Create thousands of training examples (text + nutrition labels) for machine learning

Usage: python generate_training_data.py
Input: nutrition_database.db
Output: training_data.json, training_examples.txt
"""

import sqlite3
import random
import json
import pandas as pd
import os

def load_foods_from_database():
    """
    Load all foods from the nutrition database
    """
    if not os.path.exists('nutrition_database.db'):
        print("Error: nutrition_database.db not found!")
        print("Please run 'extract_json_to_database.py' first.")
        return []
    
    conn = sqlite3.connect('nutrition_database.db')
    cursor = conn.cursor()
    
    # Get all foods that have nutrition data
    cursor.execute('''
        SELECT food_name, calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g 
        FROM foods 
        WHERE calories_per_100g > 0
    ''')
    
    foods = cursor.fetchall()
    conn.close()
    
    print(f"Loaded {len(foods)} foods with nutrition data")
    return foods

def generate_training_examples(foods, examples_per_food=5):
    """
    Generate training examples from the foods database
    """
    print(f"Generating {examples_per_food} examples per food...")
    
    training_examples = []
    
    for food_name, cal_100g, prot_100g, fat_100g, carb_100g in foods:
        
        # Skip foods with no nutrition data
        if cal_100g <= 0 and prot_100g <= 0:
            continue
            
        for _ in range(examples_per_food):
            # Generate random quantity between 50-500 grams
            quantity = round(random.uniform(50, 500), 1)
            
            # Create input text in different formats
            text_variations = [
                f"{quantity} grams of {food_name}",
                f"{quantity}g of {food_name}",
                f"{quantity} g {food_name}",
                f"{int(quantity)} grams of {food_name}" if quantity == int(quantity) else f"{quantity} grams of {food_name}"
            ]
            
            # Choose random text variation
            input_text = random.choice(text_variations)
            
            # Calculate nutrition values scaled by quantity
            scale_factor = quantity / 100  # Convert per-100g to per-quantity
            
            nutrition_labels = {
                'calories': round(cal_100g * scale_factor, 2),
                'protein': round(prot_100g * scale_factor, 2),
                'fat': round(fat_100g * scale_factor, 2),
                'carbs': round(carb_100g * scale_factor, 2)
            }
            
            # Create training example
            training_example = {
                'input_text': input_text,
                'quantity': quantity,
                'food_name': food_name,
                'nutrition': nutrition_labels
            }
            
            training_examples.append(training_example)
    
    print(f"Generated {len(training_examples)} training examples")
    return training_examples

def save_training_data(training_examples):
    """
    Save training data in multiple formats for easy inspection
    """
    # Save as JSON
    with open('training_data.json', 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    # Save as readable text file
    with open('training_examples.txt', 'w') as f:
        f.write("Training Examples for Food Nutrition ML Model\\n")
        f.write("="*60 + "\\n\\n")
        
        for i, example in enumerate(training_examples[:20]):  # First 20 examples
            f.write(f"Example {i+1}:\\n")
            f.write(f"Input: {example['input_text']}\\n")
            f.write(f"Output: {example['nutrition']['calories']:.1f} kcal, ")
            f.write(f"{example['nutrition']['protein']:.1f}g protein, ")
            f.write(f"{example['nutrition']['fat']:.1f}g fat, ")
            f.write(f"{example['nutrition']['carbs']:.1f}g carbs\\n")
            f.write("-" * 40 + "\\n")
        
        f.write(f"\\n... and {len(training_examples) - 20} more examples\\n")
    
    print("Training data saved:")
    print("- training_data.json (for machine learning)")
    print("- training_examples.txt (human readable)")

def analyze_training_data(training_examples):
    """
    Print statistics about the generated training data
    """
    print("\\n" + "="*50)
    print("TRAINING DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    total_examples = len(training_examples)
    unique_foods = len(set(ex['food_name'] for ex in training_examples))
    
    print(f"Total training examples: {total_examples:,}")
    print(f"Unique food items: {unique_foods:,}")
    print(f"Average examples per food: {total_examples/unique_foods:.1f}")
    
    # Nutrition value ranges
    calories = [ex['nutrition']['calories'] for ex in training_examples]
    proteins = [ex['nutrition']['protein'] for ex in training_examples]
    
    print(f"\\nNutrition value ranges:")
    print(f"Calories: {min(calories):.1f} - {max(calories):.1f} kcal")
    print(f"Protein: {min(proteins):.1f} - {max(proteins):.1f} g")
    
    # Sample examples
    print(f"\\nSample training examples:")
    sample_examples = random.sample(training_examples, min(5, len(training_examples)))
    
    for i, example in enumerate(sample_examples, 1):
        print(f"{i}. {example['input_text']}")
        nut = example['nutrition']
        print(f"   → {nut['calories']:.1f} kcal, {nut['protein']:.1f}g protein")

def main():
    """
    Main function to generate training data
    """
    print("="*60)
    print("Training Data Generator")
    print("="*60)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Load foods from database
    foods = load_foods_from_database()
    
    if not foods:
        print("No foods found in database!")
        return
    
    # Generate training examples
    training_examples = generate_training_examples(foods, examples_per_food=3)
    
    if not training_examples:
        print("No training examples generated!")
        return
    
    # Save training data
    save_training_data(training_examples)
    
    # Analyze the data
    analyze_training_data(training_examples)
    
    print(f"\\n✅ Success! Generated {len(training_examples)} training examples")
    print("\\nNext step: Run 'train_model.py'")

if __name__ == "__main__":
    main()