from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


num_classes = 12  # 클래스의 개수에 맞게 설정해주세요

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2, 'test_data/train', 'image', 'label', data_gen_args, num_classes)

model = unet(num_class=num_classes)
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("test_data/test", results)