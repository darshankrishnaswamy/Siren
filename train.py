import code
from keras import Sequential, optimizers
from keras.layers import Conv2D, Dropout, Flatten, Dense


def compSuperFast():
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=1), metrics=["acc"])


def compFast():
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.01), metrics=["acc"])


def comp():
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])


def askComp():
    try:
        a = int(input("\nSpeed of fitting? (3 is default, 1 is slow, 5 is fastest)\n"))
        if a == 3:
            comp()
        elif a == 4:
            compFast()
        elif a == 5:
            compSuperFast()
        else:
            print("Not implemented yet")
    except:
        print("Out of bounds")


def reset():
    global model
    model = Sequential()
    #
    filter_size = 3
    input_shape = (20, 431, 1)
    #
    model.add(Conv2D(64, kernel_size=filter_size, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Conv2D(24, kernel_size=filter_size, activation="relu"))
    model.add(Dropout(0.5))
    # model.add(Conv2D(12, kernel_size=filter_size, activation="relu"))
    model.add(Flatten())
    # model.add(Dense(16))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(50, activation="softmax"))
    #
    askComp()
    #
    a = input("Run?\n")
    if a != "n" and a != "N" and a != "":
        run()


history = 0;


def run():
    try:
        model.fit(X, Y, validation_data=(x, y), epochs=10000, batch_size=5)
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
    model.save_weights("./models/" + str(name) + ".h5")
    model_json = model.to_json()
    with open("./models/" + str(name) + ".json", "w") as json_file:
        json_file.write(model_json)


reset()

variables = globals().copy()
variables.update(locals())
shell = code.InteractiveConsole(variables)
# shell.runsource("run()")
shell.interact()
