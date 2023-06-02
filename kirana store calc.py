sum = 0
first_input = True

while True:
    Product_value = input("Enter product value or press q to exit: \n")
    if Product_value == 'q':
        if first_input:
            experience = input("Rate us from 1 to 10: ")
            print(f"Thank you for rating us {experience}!")
        break
    else:
        sum += int(Product_value)
        print(f"Product value is: {sum}")
        first_input = False

print(f"The total bill amount is {sum}")
print("Thanks for shopping")


# in this code if the first entered values are numbers then the bill
# is continued and if the customer presses q on the first input then
# only the code asks for rating