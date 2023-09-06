import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


label = []
with open('./annotations/Charades_v1_classes.txt', 'r') as f:
    for line in f.readlines():
        label.append(line.strip().split()[0])

relationship_classes = []
with open('./annotations/relationship_classes.txt', 'r') as f:
    for line in f.readlines():
        relationship_classes.append(line.strip())
relationship_classes[0] = 'looking_at'
relationship_classes[1] = 'not_looking_at'
relationship_classes[5] = 'in_front_of'
relationship_classes[7] = 'on_the_side_of'
relationship_classes[10] = 'covered_by'
relationship_classes[11] = 'drinking_from'
relationship_classes[13] = 'have_it_on_the_back'
relationship_classes[15] = 'leaning_on'
relationship_classes[16] = 'lying_on'
relationship_classes[17] = 'not_contacting'
relationship_classes[18] = 'other_relationship'
relationship_classes[19] = 'sitting_on'
relationship_classes[20] = 'standing_on'
relationship_classes[25] = 'writing_on'

object_classes = ['__background__']
with open('./annotations/object_classes.txt', 'r') as f:
    for line in f.readlines():
        object_classes.append(line.strip())
object_classes[9] = 'closet/cabinet'
object_classes[11] = 'cup/glass/bottle'
object_classes[23] = 'paper/notebook'
object_classes[24] = 'phone/camera'
object_classes[31] = 'sofa/couch'


def encode_sg(filename, object_classes, relationship_classes, MAX_SEQUENCE_LENGTH):
    x = []
    with open(filename, 'r') as sg_file:
        sgs = sg_file.readlines()
        for sg in sgs:
            v_x = []
            for sg in sg.strip().split(";"):
                f_x = np.zeros([len(relationship_classes), len(object_classes)], np.float16)
                orps = sg.split(",")
                for orp in orps:
                    if orp.replace("\n", "").strip() != "":
                        o_ = orp.split(":")[0]
                        sr_ = orp.split(":")[1].split("/")[0]
                        cr_ = orp.split(":")[1].split("/")[1]
                        f_x[relationship_classes.index(sr_), object_classes.index(o_)] = 1
                        f_x[relationship_classes.index(cr_), object_classes.index(o_)] = 1
                v_x.append(f_x)
            x.append(v_x)
    x = np.asarray(x)
    x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)
    return x


def encode_actions(filename, label):
    y = []
    with open(filename, 'r') as actions_file:
        al = actions_file.readlines()
        for actions in al:
            f_y = [0 for _ in label]
            for v in actions.strip().split(" "):
                if v != "":
                    f_y[label.index(v)] = 1
            y.append(f_y)
    y = np.asarray(y)
    return y


def create_dataset():
    MAX_SEQUENCE_LENGTH = 4
    x = encode_sg("relation_train_x.txt", object_classes, relationship_classes, MAX_SEQUENCE_LENGTH)
    y = encode_actions("action_train_y.txt", label)
    l = len(label)
    return x, y, l


train_X, train_y, l = create_dataset()

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2]*train_X.shape[3])
train_X = train_X.astype('float16')

train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=1)

model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(l, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary()) 

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(val_X, val_y), verbose=2, shuffle=False, callbacks=[early_stop])

model.save("action_anticipation_model.h5")