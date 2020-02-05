import pickle

# Load mode dari file
filename = "trained-model-kon8-10-naivebayes.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)

#contoh default data
default = [[ 0.087, 0.030, 0.024, 10.756, 0.097 ]]
print("contoh data default")
print(default)

#baca input
x1 = input("Enter x1: ") 
x2 = input("Enter x2: ") 
x3 = input("Enter x3: ") 
x4 = input("Enter x4: ") 
x5 = input("Enter x5: ") 

Xuji = [[ float(x1), float(x2), float(x3), float(x4), float(x5) ]]

#prediksi dari model
print(model.predict(Xuji))