import code
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Dropout, Flatten, Dense

history = 0;
batch_size = 5;


# def compSuperFast():
#     model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=1), metrics=["acc"])
#
#
# def compFast():
#     model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.01), metrics=["acc"])
#

def comp():
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])


# def askComp():
#     try:
#         a = int(input("\nSpeed of fitting? (3 is default, 1 is slow, 5 is fastest)\n"))
#         if a == 3:
#             comp()
#         elif a == 4:
#             compFast()
#         elif a == 5:
#             compSuperFast()
#         else:
#             print("Not implemented yet")
#     except:
#         print("Out of bounds")


def reset():
    global model
    model = Sequential()
    #
    filter_size = 3
    input_shape = (513, 27, 1)
    #
    model.add(Conv2D(48, kernel_size=filter_size, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Conv2D(20, kernel_size=filter_size, activation="relu"))
    model.add(Dropout(0.5))
    # model.add(Conv2D(12, kernel_size=filter_size, activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(20, activation="softmax"))
    #
    # askComp()
    comp()
    #
    a = input("Run?\n")
    if a != "n" and a != "N" and a != "":
        run()


def run():
    try:
        name = input("Name:\n")
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('./models/'+str(name)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4,
        #                                    mode='min')

        model.fit(X, Y, batch_size=batch_size, epochs=100,
                  callbacks=[earlyStopping, mcp_save], validation_data=(x, y))
        # model.fit(X, Y, validation_data=(x, y), epochs=10000, batch_size=5)
        # print(model.get_weights())
        # while 1:
        #     model.fit(X, Y, validation_data=(x, y), epochs=1, batch_size=5)
        #     print(model.get_weights())
    except:
        return


def run1e():
    global history
    try:
        history = model.fit(X, Y, validation_data=(x, y), epochs=1, batch_size=5)
    except:
        return


def run2():
    model.fit(X, Y, validation_data=(x, y), epochs=10000, batch_size=60)


def runSeq():
    model.fit(X, Y, validation_data=(x, y), epochs=50, batch_size=120)
    model.fit(X, Y, validation_data=(x, y), epochs=80, batch_size=60)


def save():
    name = input("Name:\n")
    model.save("./models/"+str(name)+".hdf5")
    # model.save_weights("./models/" + str(name) + ".h5")
    # model_json = model.to_json()
    # with open("./models/" + str(name) + ".json", "w") as json_file:
    #     json_file.write(model_json)


reset()

variables = globals().copy()
variables.update(locals())
shell = code.InteractiveConsole(variables)
# shell.runsource("run()")
shell.interact()
