

## Building

```
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build/
```

## run (dev version)

Get a training dataset. You can download either

* https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
* https://ibug.doc.ic.ac.uk/download/annotations/helen.zip

Make indexes to the dataset (why isn't this built in?) (**if you're using ibug, subsitute ibug/ for helen/trainset**):

```
find helen/trainset/ -name "*.jpg" | tee helen/trainset/images.txt 
find helen/trainset/ -name "*.pts" | tee helen/trainset/annotations.txt
```

Train the model:

```
./build/FacemarkTrain train helen/trainset src/models/facemark/model.yaml
```

Test the model:

```
./build/FacemarkTrain camera src/models/facemark/model.yaml
```


