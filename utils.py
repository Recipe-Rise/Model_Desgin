import numpy as np
import plotly.graph_objects as go


def cosine_similarity(vec1, vec2):
    """
    Returns the cosine similarity between two vectors of n dimension
    """
    denom = np.sqrt(np.sum(np.square(vec1))) * np.sqrt(np.sum(np.square(vec2)))
    return np.round(np.dot(vec1, vec2) / denom * 100, 2)


def calculate_bmr(weight, height, age, gender):
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation
    weight in kg, height in cm, age in years
    """
    if gender == "Male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    return bmr


def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure
    activity_level: 1.2 (sedentary), 1.375 (light), 1.55 (moderate), 1.725 (very active), 1.9 (extra active)
    """
    return bmr * activity_level


def get_macro_targets(tdee, goal, weight):
    """
    Calculate macronutrient targets based on fitness goal
    Returns: (protein, carbs, fats) in grams
    
    Protein per kg guidelines:
    - Weight loss: 2.2-2.4g/kg to preserve muscle
    - Muscle gain: 1.8-2.2g/kg
    - Weight gain: 1.6-2.0g/kg
    - Maintenance: 1.6-1.8g/kg
    
    Carb/Fat ratios adjust based on goals:
    - Weight loss: Lower carbs (30-35% of calories)
    - Muscle gain: Higher carbs (50-55% of calories)
    - Weight gain: High carbs (55-60% of calories)
    - Maintenance: Balanced (40-45% of calories)
    """
    
    if goal == "Loss weight":
        # Calculate adjusted TDEE for weight loss
        adjusted_tdee = tdee - 500  # 500 calorie deficit
        
        # Higher protein for muscle preservation during deficit
        protein = weight * 2.4  # 2.4g per kg
        # Lower carbs for weight loss
        carbs_calories = adjusted_tdee * 0.30  # 30% of calories from carbs
        carbs = carbs_calories / 4
        # Moderate fats
        fats_calories = adjusted_tdee * 0.35  # 35% of calories from fats
        fats = fats_calories / 9
        
    elif goal == "Loss weight and gain muscles":
        # Moderate deficit with high protein
        adjusted_tdee = tdee - 300  # 300 calorie deficit
        
        # Very high protein for muscle preservation and growth
        protein = weight * 2.6  # 2.6g per kg
        # Moderate carbs for energy during workouts
        carbs_calories = adjusted_tdee * 0.35  # 35% of calories from carbs
        carbs = carbs_calories / 4
        # Lower fats
        fats_calories = adjusted_tdee * 0.30  # 30% of calories from fats
        fats = fats_calories / 9
        
    elif goal == "Gain weight":
        # Calculate adjusted TDEE for weight gain
        adjusted_tdee = tdee + 500  # 500 calorie surplus
        
        # Moderate protein
        protein = weight * 1.8  # 1.8g per kg
        # High carbs for weight gain
        carbs_calories = adjusted_tdee * 0.55  # 55% of calories from carbs
        carbs = carbs_calories / 4
        # Moderate fats
        fats_calories = adjusted_tdee * 0.30  # 30% of calories from fats
        fats = fats_calories / 9
        
    elif goal == "Gain muscles":
        # Calculate adjusted TDEE for muscle gain
        adjusted_tdee = tdee + 300  # 300 calorie surplus
        
        # High protein for muscle growth
        protein = weight * 2.2  # 2.2g per kg
        # High carbs for energy and recovery
        carbs_calories = adjusted_tdee * 0.50  # 50% of calories from carbs
        carbs = carbs_calories / 4
        # Moderate fats
        fats_calories = adjusted_tdee * 0.25  # 25% of calories from fats
        fats = fats_calories / 9
        
    else:  # Fitness (maintain)
        # Maintenance calories
        adjusted_tdee = tdee
        
        # Moderate protein
        protein = weight * 1.8  # 1.8g per kg
        # Balanced carbs
        carbs_calories = adjusted_tdee * 0.45  # 45% of calories from carbs
        carbs = carbs_calories / 4
        # Balanced fats
        fats_calories = adjusted_tdee * 0.30  # 30% of calories from fats
        fats = fats_calories / 9

    # Calculate actual calories from macros for verification
    total_calories = (protein * 4) + (carbs * 4) + (fats * 9)
    
    # Print debug information
    print(f"\nGoal: {goal}")
    print(f"Weight: {weight}kg")
    print(f"Original TDEE: {tdee:.0f} calories")
    print(f"Adjusted TDEE: {adjusted_tdee:.0f} calories")
    print(f"Calculated calories from macros: {total_calories:.0f}")
    print("\nMacronutrient Distribution:")
    print(f"Protein: {protein:.0f}g ({(protein * 4 / total_calories * 100):.0f}%)")
    print(f"Carbs: {carbs:.0f}g ({(carbs * 4 / total_calories * 100):.0f}%)")
    print(f"Fats: {fats:.0f}g ({(fats * 9 / total_calories * 100):.0f}%)")

    return round(protein), round(carbs), round(fats)


def filter_by_macros(df, protein_target, carbs_target, fats_target, tolerance=0.5):
    """
    Filter recipes based on macronutrient targets with a tolerance
    tolerance: 50% deviation from target is allowed
    """
    # Get the nutritional information columns (indices 8:13)
    nutrition_cols = df.columns[8:13]
    print("Nutrition columns:", nutrition_cols)
    
    # Convert PDV to actual grams (assuming PDV is percentage of daily value)
    # We'll use rough estimates: protein_dv = 50g, carbs_dv = 300g, fats_dv = 65g
    protein_dv = 50
    carbs_dv = 300
    fats_dv = 65
    
    # Convert targets to PDV
    protein_pdv = (protein_target / protein_dv) * 100
    carbs_pdv = (carbs_target / carbs_dv) * 100
    fats_pdv = (fats_target / fats_dv) * 100
    
    print(f"Target PDV values: Protein={protein_pdv:.2f}%, Carbs={carbs_pdv:.2f}%, Fats={fats_pdv:.2f}%")
    
    # Apply macro filters with tolerance
    # We'll use a more lenient approach - recipes should meet at least one of the macro targets
    df_filtered = df[
        # Protein filter (using protein PDV)
        (df[nutrition_cols[3]] >= protein_pdv * (1 - tolerance)) |
        # Carbs filter (using sugar PDV as an estimate)
        (df[nutrition_cols[1]] >= carbs_pdv * (1 - tolerance)) |
        # Fats filter (using total fat PDV)
        (df[nutrition_cols[0]] >= fats_pdv * (1 - tolerance))
    ]
    
    print(f"Original recipes: {len(df)}")
    print(f"Filtered recipes: {len(df_filtered)}")
    
    # Add nutritional information to the filtered recipes
    df_filtered['protein_pdv'] = df_filtered[nutrition_cols[3]]
    df_filtered['carbs_pdv'] = df_filtered[nutrition_cols[1]]  # Using sugar PDV as an estimate
    df_filtered['fats_pdv'] = df_filtered[nutrition_cols[0]]
    
    # Sort by how well they match the targets
    df_filtered['macro_score'] = (
        (df_filtered['protein_pdv'] / protein_pdv) +
        (df_filtered['carbs_pdv'] / carbs_pdv) +
        (df_filtered['fats_pdv'] / fats_pdv)
    ) / 3
    
    df_filtered = df_filtered.sort_values('macro_score', ascending=False)
    
    return df_filtered


def filter_recipes_by_goal(df, fitness_goal, macro_targets=None):
    """
    Filter recipes based on fitness goal and macro targets.
    Prioritizes recipes that match the nutritional needs of each goal.
    """
    try:
        # Get nutrition columns
        nutrition_cols = df.columns[8:13]
        protein_target, carbs_target, fats_target = macro_targets

        # Convert nutritional values to grams
        df['protein_g'] = df[nutrition_cols[3]] * 0.5  # 50g protein is 100% DV
        df['carbs_g'] = df[nutrition_cols[1]] * 3.0    # 300g carbs is 100% DV
        df['fats_g'] = df[nutrition_cols[0]] * 0.65    # 65g fat is 100% DV

        # Calculate macro ratios per recipe
        total_cals = (df['protein_g'] * 4) + (df['carbs_g'] * 4) + (df['fats_g'] * 9)
        df['protein_ratio'] = (df['protein_g'] * 4) / total_cals
        df['carbs_ratio'] = (df['carbs_g'] * 4) / total_cals
        df['fats_ratio'] = (df['fats_g'] * 9) / total_cals

        # Score recipes based on goal
        if fitness_goal == "Loss weight":
            # Priority: High protein, low carbs, moderate fats
            df['goal_score'] = (
                (df['protein_ratio'] * 0.5) +        # 50% of score from protein content
                ((1 - df['carbs_ratio']) * 0.3) +    # 30% of score from low carbs
                ((1 - df['fats_ratio']) * 0.2)       # 20% of score from moderate fats
            )
            # Filter out very high-calorie recipes
            df = df[df['calories'] <= (protein_target * 4 + carbs_target * 4 + fats_target * 9) * 0.4]

        elif fitness_goal == "Loss weight and gain muscles":
            # Priority: Very high protein, moderate carbs, low fats
            df['goal_score'] = (
                (df['protein_ratio'] * 0.6) +        # 60% of score from protein content
                ((1 - df['carbs_ratio']) * 0.25) +   # 25% of score from moderate carbs
                ((1 - df['fats_ratio']) * 0.15)      # 15% of score from low fats
            )
            # Ensure adequate protein per serving
            df = df[df['protein_g'] >= protein_target * 0.3]  # At least 30% of daily protein target

        elif fitness_goal == "Gain weight":
            # Priority: Balanced macros, higher calories
            df['goal_score'] = (
                (df['calories'] / 1000 * 0.4) +      # 40% of score from calories
                (df['carbs_ratio'] * 0.35) +         # 35% of score from carbs
                (df['protein_ratio'] * 0.25)         # 25% of score from protein
            )
            # Filter for higher calorie recipes
            df = df[df['calories'] >= 400]

        elif fitness_goal == "Gain muscles":
            # Priority: High protein, high carbs, moderate fats
            df['goal_score'] = (
                (df['protein_ratio'] * 0.45) +       # 45% of score from protein
                (df['carbs_ratio'] * 0.45) +         # 45% of score from carbs
                ((1 - df['fats_ratio']) * 0.1)       # 10% of score from moderate fats
            )
            # Ensure good protein content
            df = df[df['protein_g'] >= protein_target * 0.25]  # At least 25% of daily protein target

        else:  # Fitness (maintain)
            # Priority: Balanced macros
            df['goal_score'] = (
                (1 - abs(df['protein_ratio'] - 0.3)) * 0.34 +  # 34% protein ratio
                (1 - abs(df['carbs_ratio'] - 0.4)) * 0.33 +    # 33% carbs ratio
                (1 - abs(df['fats_ratio'] - 0.3)) * 0.33       # 33% fats ratio
            )

        # Sort by goal score
        df = df.sort_values('goal_score', ascending=False)

        # Keep only top 60% of recipes that best match the goal
        df = df.head(int(len(df) * 0.6))

        # Clean up temporary columns
        df = df.drop(['protein_g', 'carbs_g', 'fats_g', 
                     'protein_ratio', 'carbs_ratio', 'fats_ratio', 
                     'goal_score'], axis=1)

        return df

    except Exception as e:
        print(f"Error in filter_recipes_by_goal: {str(e)}")
        print("Available columns:", df.columns.tolist())
        return df


def find_similar_recipe(
    recipe,
    df,
    num_recipes,
    min_calories=None,
    max_calories=None,
    meal_history=None,
    diabetic_friendly=False,
    macro_targets=None,
    fitness_goal=None,
    max_prep_time=None
):
    """
    Find similar recipes with goal-based filtering
    """
    if recipe in df["name"].to_list():
        index = df[df["name"] == recipe].index[0]
        data = df.iloc[index]
        vector = data["embedding"]

        # Find similar recipe
        df_result = df.copy()
        
        # Exclude recipes from meal history
        if meal_history:
            df_result = df_result[~df_result["name"].isin(meal_history)]
        
        # Calculate similarity scores
        df_result["similarity"] = df_result["embedding"].apply(
            lambda x: cosine_similarity(vector, x)
        )
        
        # Apply goal-based filtering first
        if fitness_goal and macro_targets:
            df_result = filter_recipes_by_goal(df_result, fitness_goal, macro_targets)
            print(f"Recipes after goal filtering: {len(df_result)}")

        # Apply calorie filters
        if min_calories is not None:
            df_result = df_result[df_result["calories"] >= min_calories]
        if max_calories is not None:
            df_result = df_result[df_result["calories"] <= max_calories]
            
        # Apply preparation time filter
        if max_prep_time is not None:
            df_result = df_result[df_result["minutes"] <= max_prep_time]
            print(f"Recipes after time filtering: {len(df_result)}")
        
        # Apply diabetic filter
        if diabetic_friendly:
            nutrition_cols = df_result.columns[8:13]
            df_result = df_result[df_result[nutrition_cols[1]] * 0.5 <= 25]

        # Sort by similarity among filtered results
        df_result = df_result.sort_values(by="similarity", ascending=False)
        
        # Get the requested number of recipes
        df_result = df_result.iloc[1:num_recipes + 1]
        
        # Drop the embedding column before returning
        if "embedding" in df_result.columns:
            df_result.drop("embedding", inplace=True, axis=1)

        return df_result

    return None


def get_recommendations_from_history(
    df,
    meal_history,
    num_recipes,
    min_calories=None,
    max_calories=None,
    diabetic_friendly=False,
    macro_targets=None,
    fitness_goal=None,
    max_prep_time=None
):
    """
    Get recommendations based on meal history with enhanced filtering and personalization
    """
    if not meal_history:
        print("No meal history provided")
        return None
        
    try:
        # Get the recipes from history
        history_recipes = df[df["name"].isin(meal_history)]
        if len(history_recipes) == 0:
            print("No matching recipes found in meal history")
            return None
        
        # Calculate average embedding of meal history
        avg_embedding = np.mean([recipe["embedding"] for _, recipe in history_recipes.iterrows()], axis=0)
        
        # Find similar recipes
        df_result = df.copy()
        
        # Exclude recipes already in history
        df_result = df_result[~df_result["name"].isin(meal_history)]
        print(f"Recipes after excluding history: {len(df_result)}")
        
        # Calculate base similarity scores
        df_result["similarity"] = df_result["embedding"].apply(
            lambda x: cosine_similarity(avg_embedding, x)
        )
        
        # Apply goal-based filtering
        if fitness_goal and macro_targets:
            df_result = filter_recipes_by_goal(df_result, fitness_goal, macro_targets)
            print(f"Recipes after goal filtering: {len(df_result)}")
        
        # Apply calorie filters
        if min_calories is not None:
            df_result = df_result[df_result["calories"] >= min_calories]
        if max_calories is not None:
            df_result = df_result[df_result["calories"] <= max_calories]
        print(f"Recipes after calorie filtering: {len(df_result)}")
        
        # Apply preparation time filter
        if max_prep_time is not None:
            df_result = df_result[df_result["minutes"] <= max_prep_time]
            print(f"Recipes after time filtering: {len(df_result)}")
        
        # Apply diabetic filter if needed
        if diabetic_friendly:
            nutrition_cols = df_result.columns[8:13]
            df_result = df_result[df_result[nutrition_cols[1]] * 0.5 <= 25]
            print(f"Recipes after diabetic filtering: {len(df_result)}")
        
        # Calculate additional preference scores based on history
        history_ingredients = set()
        history_cuisines = set()
        total_rating = 0
        n_rated = 0
        
        for _, recipe in history_recipes.iterrows():
            history_ingredients.update(recipe["ingredients"])
            # Extract cuisine from recipe name (simple implementation)
            cuisine = recipe["name"].split()[0].lower()
            history_cuisines.add(cuisine)
            
            # Get recipe rating if available
            if hasattr(recipe, "rating"):
                total_rating += recipe.rating
                n_rated += 1
        
        # Calculate average rating
        avg_rating = total_rating / n_rated if n_rated > 0 else 0
        
        # Score recipes based on multiple factors
        df_result["ingredient_score"] = df_result["ingredients"].apply(
            lambda x: len(set(x) & history_ingredients) / len(set(x))
        )
        
        # Cuisine similarity score
        df_result["cuisine_score"] = df_result["name"].apply(
            lambda x: 1 if x.split()[0].lower() in history_cuisines else 0
        )
        
        # Combine all scores with weights
        df_result["final_score"] = (
            df_result["similarity"] * 0.4 +           # Base similarity
            df_result["ingredient_score"] * 0.3 +     # Ingredient overlap
            df_result["cuisine_score"] * 0.2 +        # Cuisine match
            (df_result["rating"] / 5) * 0.1          # Rating (if available)
        )
        
        # Sort by final score
        df_result = df_result.sort_values("final_score", ascending=False)
        
        # Get top recommendations
        df_result = df_result.iloc[:num_recipes]
        
        # Clean up temporary columns
        df_result = df_result.drop(["similarity", "ingredient_score", "cuisine_score", 
                                  "final_score", "embedding"], axis=1)
        
        print(f"Final number of recommendations: {len(df_result)}")
        return df_result
        
    except Exception as e:
        print(f"Error in get_recommendations_from_history: {str(e)}")
        return None


def setColor(pdv):
    if pdv < 5:
        return "red"

    elif pdv >= 5 and pdv < 20:
        return "red"

    elif pdv > 20:
        return "red"


def plot_nutrition(dfx):
    # Get the nutrition columns (indices 8:13)
    nutrition_cols = dfx.index[8:13]
    
    # Convert PDV to grams using standard daily values
    # Standard daily values: protein=50g, carbs=300g, fats=65g, sugar=50g
    protein_g = dfx[nutrition_cols[3]] * 0.5  # Convert PDV to grams
    carbs_g = dfx[nutrition_cols[1]] * 3  # Convert PDV to grams (using sugar as estimate for carbs)
    fats_g = dfx[nutrition_cols[0]] * 0.65  # Convert PDV to grams
    sugar_g = dfx[nutrition_cols[1]] * 0.5  # Convert PDV to grams
    
    # Create the bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Protein", "Carbs", "Fats", "Sugar"],
                y=[protein_g, carbs_g, fats_g, sugar_g],
                text=[f"{protein_g:.1f}g", f"{carbs_g:.1f}g", f"{fats_g:.1f}g", f"{sugar_g:.1f}g"],
                textposition="auto",
                marker_color=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"],
            )
        ]
    )
    
    # Update layout
    fig.update_layout(
        title="Nutritional Information (per serving)",
        xaxis_title="Nutrients",
        yaxis_title="Grams",
        showlegend=False,
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Update axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    return fig
