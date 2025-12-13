import traceback

def calculator():
    age = input("Enter the dog's age in dog years: ")

    try:
        d_age = float(age)

        if d_age < 0:
            print(f"The given dog age is negative. Please enter a positive age.")
            return
        elif d_age == 0:
            print(f"The given dog age is 0. Please enter a positive age.")
            return

        human_age = 0

        if d_age >= 5:
            human_age += 5 * 7.2
            remaining_years = d_age - 5
            human_age += remaining_years * 7
        else:
            human_age = d_age * 7.2
        
        rounded_age = round(human_age, 2)
        
        if rounded_age == int(rounded_age):
            formatted_age = f"{rounded_age:.1f}"
        else:
            formatted_age = f"{rounded_age:.2f}"

        print(f"The given dog age {d_age} is {formatted_age} in human years.")

    except ValueError:
        print(f"'{age}' is an invalid age. Input must be numeric and positive.")

calculator() # This line calls the calculator function