# Image Classification In SIKS-GIS

<aside>
🧑🏽‍🏫 **Daftar isi**: Latar Belakang, Dasar Teori, Implementasi, Data Preparation, **Modelling with CNN,** **Modelling with VGG16**, Evaluasi

</aside>

# Latar Belakang

## SIKS-GIS

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e472f02-567b-416a-83bd-69c3ed3d5b25/Untitled.png)

### “***Aplikasi survey lapangan kepada Keluarga Penerima Manfaat (KPM) yang berisi 9 kuesioner dan 1 foto rumah”***

![Download aplikasi di Playstore](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f64f1476-6bce-4f7a-abe7-406f99726df0/Untitled.png)

Download aplikasi di Playstore

![Tampilan aplikasi](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ab027187-f3f4-4f51-a1f7-e7d04e1a1982/Untitled.png)

Tampilan aplikasi

## Tujuan Aplikasi

1. Validasi kondisi terkini tiap penerima KPM,
2. Memberikan rekomendasi kelayakan penerimaan bansos bagi KPM.

## Kondisi Terikini

- Terdapat **1,7 juta** data masuk seluruh Indonesia dengan **200 ribu** lebih data yang sudah diverifikasi secara manual oleh arahan Menteri Sosial

## Permasalahan

- Masukan/input data kuesioner memiliki validasi yang rendah atau **tidak akurat**
- Foto Rumah memiliki **konsistensi yang rendah,** namun lebih valid dibanding data input kuesioner
- Verifikasi manual membutuhkan waktu yang **lama** dan rawan terjadinya **inkonsistensi**

## Tujuan Project:

- **Mengembangkan model** klasifikasi kelayakan penerimaan bansos berdasarkan foto rumah
- **Memberikan rekomendasi** penilaian untuk verifikator

# **Dasar Teori**

## 1. Python

Python adalah salah satu bahasa pemrograman paling populer di dunia. Python adalah bahasa pemrograman yang terinterpretasi, object oriented dan high level. Penggunaan Python dalam Data Science antara lain,

- Data collection & cleaning
- Data Exploration
- Data Visualization & Interpretation
- Data Modelling
- Deploying

## 2. Open CV

OpenCV (Open Source Computer Vision Library), adalah sebuah library open source yang dikembangkan oleh intel yang fokus untuk menyederhanakan programing terkait citra digital. fitur utama dari OpenCV antara lain :

- Image and video I/O
- Computer Vision secara umum dan pengolahan citra digital
- Modul computer vision high level
- Metode untuk AI dan machine learning
- Sampling gambar dan transformasi

## 3. CNN

Convolutional Neural Network (CNN) adalah salah satu jenis neural network yang biasa digunakan pada data image. CNN bisa digunakan untuk mendeteksi dan mengenali object pada sebuah image.

CNN memanfaatkan proses konvolusi dengan menggerakan sebuah kernel konvolusi (filter) berukuran tertentu ke sebuah gambar, komputer mendapatkan informasi representatif baru dari hasil perkalian bagian gambar tersebut dengan filter yang digunakan.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/95449de6-b4ad-41f7-91c6-3dd95d18ce05/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/22872856-7e0f-4da1-ac20-865526a4c5ec/Untitled.png)

## 4. VGG16

VGG16 merupakan model CNN yang memanfaatkan convolutional layer dengan spesifikasi convolutional filter yang kecil (3×3). Dengan ukuran convolutional filter tersebut, kedalaman neural network dapat ditambah dengan lebih banyak lagi convolutional layer.

Model VGG16 mempunyai 19 layer yang terdiri dari 16 convolutional layer dan 3 fully-connected layer

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/10060cc2-de92-448e-8c58-f98ed12226c7/Untitled.png)

# Implementasi: Data Preparation

## Scrapping Data

Diambil melalui website: [https://siksgis.kemensos.go.id/](https://siksgis.kemensos.go.id/)

Data diambil dengan scrapping HTML

```python
def download_image(url, start, label):
    myFile=open(url,'r')
    soup = BeautifulSoup(myFile, "html.parser")
    img = soup.find_all("img")
    file_name = start
    for i in img:
        img_data = requests.get(i['src']).content
        with open('img/'+label+'/'+str(file_name)+'.jpg', 'wb') as handler:
            handler.write(img_data)
        file_name += 1
download_image('url-local-tidak-layak.html', 1, 'tidak-layak')
```

## Read and Preprocessing Data

```python
def load_data():
    """
        Load the data:
            - 5,400 images to train the network.
            - 2,298 images to evaluate how accurately the network learned to classify images.
    """
    
    datasets = ['../1_WrappingData/img/data_train', '../1_WrappingData/img/data_test']
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
```

### Image Augmentation

**Menambahkan variasi** dataset menjadi 3x lipat lebih banyak

```python
def fill(img, h, w):
    return cv2.resize(img, (h, w), cv2.INTER_CUBIC)

def augmentation(img_src):
# Horizontal flip
    img = img_src
    if(random.randint(0, 1)):
        img = cv2.flip(img, 1)

# Rotation
    angle = int(random.uniform(-90, 90))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

# Zoom
    value = random.uniform(0.6, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img_con = img_src[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img_con, h, w)

# brighness
    value = random.uniform(0.5, 1.4)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img
```

```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (**150, 150**))
image_aug = augmentation(image)
```

```python
train_images = train_images / 255.0 
test_images = test_images / 255.0
```

### Proportion of each observed category

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3b16d2e0-7489-41f9-ba2f-57c1d19f3a6e/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fecb7881-0386-4c4a-b004-af4edce1099c/Untitled.png)

### Some examples of image of the dataset

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2c713597-acbb-4bc7-a44e-a2430a0cca58/Untitled.png)

### Questionnaire data

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b093a512-b337-416d-b502-cfb1a936c063/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef17f0bd-fcf4-4c00-aaf2-dd6282f6d3be/Untitled.png)

# Modeling with CNN

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70c548f3-c298-464b-9be8-d6b48ae91706/Untitled.png)

Steps are:

1. Build the model,
2. Compile the model,
3. Train / fit the data to the model,
4. Evaluate the model on the testing set,
5. Carry out an error analysis of our model.

layers explanation:

- Conv2D: (32 filters of size 3 by 3) The features will be "extracted" from the image.
- MaxPooling2D: The images get half sized.
- Flatten: Transforms the format of the images from a 2d-array to a 1d-array of 150 150 3 pixel values.
- Relu : given a value x, returns max(x, 0).
- Softmax: 6 neurons, probability that the image belongs to one of the classes.

## Build the model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3), padding='same'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
```

## Compile the model

Then, we can compile it with some parameters such as:

- **Optimizer**: adam = RMSProp + Momentum.
- Momentum = takes into account past gradient to have a better update.
- RMSProp = exponentially weighted average of the squares of past gradients.
- **Loss function**: we use sparse binary crossentropy for classification, due to `layak` or `tidak-layak`

```python
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
```

## Train / fit the data to the model

```python
history = model.fit(train_images, train_labels, batch_size=32, epochs=10, 
										validation_split = 0.2)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4ae8ea1c-8fc5-4bd7-a063-6673c01049ce/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1df00c8-cdb0-4cc6-a1ac-5bffb6283206/Untitled.png)

## Confusion Matrix

![Confusion Matrix on Validation image dataset](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61ac85bd-d050-4bcb-8064-73066dcd842f/Untitled.png)

Confusion Matrix on Validation image dataset

![Confusion Matrix on Train image dataset](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/28fe0de6-53ad-446b-a3be-5319c39ee9cd/Untitled.png)

Confusion Matrix on Train image dataset

## Evaluate the model on the testing set

```python
predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = ((predictions > 0.5)+0).ravel() # We take the highest probability

print("pred_labels\n", pred_labels)
print("test_labels\n", test_labels)
```

```
pred_labels
 [0 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 0 0 1 0 0 1 0 1 1 0 0 1 0 1 1 1 1
 1 0 0 1 1 0 0 0 0 1 1 1 1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 0 0 0
 1 0 0 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0]
test_labels
 [1 0 1 1 1 0 0 0 0 0 0 0 1 0 0 1 1 0 0 1 1 1 1 0 0 1 1 1 1 0 0 1 0 1 0 1 0
 1 0 0 0 1 0 0 1 1 0 1 1 1 0 1 0 1 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 1 0 0 0
 1 0 1 0 1 1 1 1 0 1 1 0 1 0 1 0 1 1 0 0 0 1 0]
```

```python
print_predicted_images(class_names, test_images, test_labels, pred_labels)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7f8b9573-d90e-4564-818c-ac3d364c4ff1/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8213cfcf-1628-4982-8598-d0b09aed1754/Untitled.png)

# Modeling with VGG16

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0752c937-7d09-40de-a5c5-8da12eaf2878/Untitled.png)

Steps are:

1. Build the model,
2. Compile the model,
3. Train / fit the data to the model,
4. Evaluate the model on the testing set,
5. Carry out an error analysis of our model.

## Build the model

```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

model_vgg = VGG16(weights='imagenet', include_top=False)

model_vgg = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (x, y, z)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
```

## Compile the model

Then, we can compile it with some parameters such as:

- **Optimizer**: adam = RMSProp + Momentum.
- Momentum = takes into account past gradient to have a better update.
- RMSProp = exponentially weighted average of the squares of past gradients.
- **Loss function**: we use sparse binary crossentropy for classification, due to `layak` or `tidak-layak`
- **ModelCheckpoint**: Save or checkpoint the model

```python
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history2 = model2_vgg.fit(train_features, train_labels, batch_size=64, epochs=15, 
													validation_split = 0.2, callbacks=[cp_callback])
```

## Train / fit the data to the model

```python
history = model.fit(train_images, train_labels, batch_size=32, epochs=10, 
										validation_split = 0.2)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/47e2ceaa-d532-4f12-8cdd-b140a37d17f2/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/358a814b-5988-4373-a28a-80ec4888774a/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1050203f-cbbd-4668-8497-8a551ffb54f5/Untitled.png)

Validation Accuracy: **79,44%**

Testing Accuracy: **64,75%**

NB: Pada data testing masih banyak foto rumah yang tidak ideal atau tidak jelas

## Evaluate the model on the testing set

```python
predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = ((predictions > 0.45)+0).ravel() # We take the highest probability

print("pred_labels\n", pred_labels)
print("test_labels\n", test_labels)
```

```python
display_examples(class_names, test_images, pred_labels)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/927e99f3-62d7-4632-9331-098ac50e7205/Untitled.png)

Membuat ground truth

## Error analysis

We can try to understand on which kind of images the classifier has trouble.

```python
def print_predicted_images(class_names, test_images, test_labels, pred_labels):
    """
        Print 25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels
    """
    BOO = (test_labels == pred_labels)
    mislabeled_indices = np.where(BOO == 0)
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]

    title = "Some examples of mislabeled images by the classifier:"
    display_examples(class_names,  mislabeled_images, mislabeled_labels, title)
    
    correct_indices = np.where(BOO == 1)
    correct_images = test_images[correct_indices]
    correct_labels = pred_labels[correct_indices]

    title = "Some examples of correct images by the classifier:"
    display_examples(class_names,  correct_images, correct_labels, title)

print_predicted_images(class_names, test_images, test_labels, pred_labels)
```

### Data Prediksi Salah

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0af04625-367f-4c8b-ab00-a8d2c79b6747/Untitled.png)

### Data Prediksi Benar

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e0d1063e-a265-497c-a23b-0716fde9dea9/Untitled.png)

### Berdasarkan data prediksi benar…

Karakteristik foto rumah yang **tidak layak** mendapatkan bansos antara lain:

- Memiliki objek furniture rumah yang banyak
- Tone foto yang lebih cerah dan ngejreng
- Memiliki cat rumah berwarna terang dan bermacam macam

Sedangkan karakteristik foto rumah yang **layak** antara lain:

- Simple, sederhana, tidak banyak objek (tembok saja atau pintu saja)
- Tone foto Cenderung gelap dan suram
- Warna rumah yang tidak dicat (abu-abu bata, merah bata, cokelat kayu, krem rotan, dll)

### Confution Matrix

![Confusion matrix with **validation image**](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5d807a3-99f4-4aef-8eb5-612e53a356fc/Untitled.png)

Confusion matrix with **validation image**

![Confusion matrix with **train image**](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/380b971c-05bd-4fef-8b10-0557e30ecf5b/Untitled.png)

Confusion matrix with **train image**

## VGG16 Without stretch image dataset (Uji Coba)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11753c21-d2bf-440b-9aa6-9a87a511bb24/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/18b36fe9-d034-4530-a517-929f0f47391b/Untitled.png)

# Masukan dan Saran (Presentasi 01/08/2022)

- Memberi batasan masalah. Di luar batasan masalah tidak dihitung
- Questionnaire data Tanpa NIK
- Narasi laporan, masih ada beberapa miss spelling. SIKS GIS tidak hanya PKH
- Uji coba manual
- Pengecekan dataset
- Memfilter sample yang sesuai kriteria, Memperbanyak sampel
- Bisa dikembagnkan lagi, label yang layak (ekstrem, pemberdayaan, dkk)
- Server, transfer knowledge, running 6 TB data, koordinasi dengan Mas Tyo