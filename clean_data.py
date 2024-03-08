"""Taller evaluable presencial"""

import nltk
import pandas as pd


def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""
    df = pd.read_csv(input_file)
    return df


def create_fingerprint(df):
    """Cree una nueva columna en el DataFrame que contenga el fingerprint de la columna 'text'"""

    df = df.copy()
    df["key"] = df["text"] #crear una columna llamada key con los valores de la columna text
    df["key"] = df["key"].str.strip() #quita los espacios en blanco del inicio y del final
    df["key"] = df["key"].str.lower() # minusculas
    df["key"] = df["key"].str.replace("-", "") #reemplaza - por nada
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    ) #quita todos los signos
    df["key"] = df["key"].str.split() # divide las palabras en una lista
    stemmer = nltk.PorterStemmer() # 
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])#aplica a cada elemento la 
    #funcion lambda cuyo argumento es x con un comprehension (ciclo for) y para cada palabra
    #apliquele el stem (convierte las palabras en su palabra base ej: applications = applic) 
    df["key"] = df["key"].apply(lambda x: sorted(set(x))) # sorted ordena alfabeticamente y set para no repetir elementos
    df["key"] = df["key"].str.join(" ") #unirlos por cadena

    return df


def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    

    # 1. Ordene el dataframe por 'fingerprint' y 'text'
    # 2. Seleccione la primera fila de cada grupo de 'fingerprint'
    # 3.  Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    # 4. Cree la columna 'cleaned' usando el diccionario
    df = df.copy()
    df = df.sort_values(by=["key", "text"], ascending=[True, True])
    keys = df.drop_duplicates(subset="key", keep="first")
    key_dict = dict(zip(keys["key"], keys["text"]))
    df["cleaned"] = df["key"].map(key_dict)

    return df




def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    # Solo contiene una columna llamada 'texto' al igual
    # que en el archivo original pero con los datos limpios
    df = df.copy()
    df = df[["cleaned"]]
    df = df.rename(columns={"cleaned": "text"})
    df.to_csv(output_file, index=False)

def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )
