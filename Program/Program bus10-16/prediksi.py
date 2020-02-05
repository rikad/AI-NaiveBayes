import pickle

# Load mode dari file
filename = "trained-model-bus10-16-naivebayes.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)

#contoh default data
default = [[ 0.087, 0.030, 0.024, 10.756 ]]
print("contoh data default")
print(default)

#baca input
x1 = input("Enter x1: ")
x2 = input("Enter x2: ")
x3 = input("Enter x3: ")
x4 = input("Enter x4: ")

Xuji = [[ float(x1), float(x2), float(x3), float(x4) ]]

#prediksi dari model
print(model.predict(Xuji))