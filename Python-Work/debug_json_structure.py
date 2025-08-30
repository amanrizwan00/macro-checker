#!/usr/bin/env python3
"""
Debug Script: Examine JSON Structure
Purpose: Look at the actual structure of your USDA JSON file to understand the format

Usage: python debug_json_structure.py
"""

import json
import os

def debug_json_structure(filename):
    """
    Debug the JSON file structure to understand the data format
    """
    print(f"Debugging JSON file: {filename}")
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
            # Check if it's already a JSON array
            if content.startswith('[') and content.endswith(']'):
                data = json.loads(content)
                print("File is a JSON array")
            else:
                # Try to make it a JSON array
                content = content.rstrip(',')
                json_content = '[' + content + ']'
                data = Json.loads(json_content)
                print("Converted to JSON array")
                
        print(f"Total items in JSON: {len(data)}")
        
        if len(data) > 0:
            print("\n" + "="*60)
            print("EXAMINING FIRST FOOD ITEM")
            print("="*60)
            
            first_item = data[0]
            
            # Print all top-level keys
            print("Top-level keys in first item:")
            for key in first_item.keys():
                print(f"  - {key}: {type(first_item[key])}")
            
            # Look for description
            if 'description' in first_item:
                print(f"\nDescription: {first_item['description']}")
            
            # Look for foodNutrients structure
            if 'foodNutrients' in first_item:
                nutrients = first_item['foodNutrients']
                print(f"\nFood nutrients found: {len(nutrients)} nutrients")
                
                if len(nutrients) > 0:
                    print("\nFirst 3 nutrients structure:")
                    for i, nutrient in enumerate(nutrients[:3]):
                        print(f"\nNutrient {i+1}:")
                        for key, value in nutrient.items():
                            print(f"  {key}: {value}")
                        
                        # Look deeper into nutrient structure
                        if 'nutrient' in nutrient:
                            print("  nutrient details:")
                            for key, value in nutrient['nutrient'].items():
                                print(f"    {key}: {value}")
            
            # Look for other possible nutrient containers
            possible_nutrient_keys = [key for key in first_item.keys() if 'nutrient' in key.lower()]
            if possible_nutrient_keys:
                print(f"\nOther possible nutrient keys: {possible_nutrient_keys}")
            
            print("\n" + "="*60)
            print("FULL STRUCTURE OF FIRST ITEM")
            print("="*60)
            
            # Print the full structure (truncated for readability)
            import json
            print(json.dumps(first_item, indent=2)[:2000] + "...")
            
    except Exception as e:
        print(f"Error examining JSON: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Update this path to match your file location
    json_filename = '../Raw-data/FoodData_Central_foundation_food_json_2025-04-24.json'
    debug_json_structure(json_filename)

if __name__ == "__main__":
    main()