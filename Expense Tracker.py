expenses = []

def add_expense():
    try:
        name = input("Enter expense name: ").strip()
        amount = float(input("Enter amount: "))
        expenses.append((name, amount))
        print("✔ Expense added successfully.")
    except ValueError:
        print("❌ Invalid amount! Please enter a number.")

def view_expenses():
    if not expenses:
        print("No expenses recorded yet.")
        return

    total = 0
    print("\n--- Expense List ---")
    for name, amount in expenses:
        print(f"{name}: ₹{amount}")
        total += amount

    print("----------------------")
    print("Total Expenses: ₹", total)

def menu():
    print("\n1. Add Expense")
    print("2. View Expenses")
    print("3. Exit")

while True:
    menu()
    choice = input("Choose an option: ").strip()

    if choice == '1':
        add_expense()
    elif choice == '2':
        view_expenses()
    elif choice == '3':
        print("Exiting program. Goodbye!")
        break
    else:
        print("Invalid choice! Please select 1, 2, or 3.")