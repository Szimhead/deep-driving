from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from EMU import EMAU

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

def build_emu(input_shape, num_classes):
    #inputs = Input(shape=input_shape,batch_size=12)
    inputs = Input(shape=input_shape)
    # Add your EMU block here
    emu_block = EMAU(c=256,k=64, stage_num=3)
    b1_with_emu, _ = emu_block(inputs)
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(b1_with_emu)

    model = Model(inputs, outputs, name="EMU")
    return model

def build_unet_with_emu(input_shape, num_classes):
    #inputs = Input(shape=input_shape,batch_size=12)
    inputs = Input(shape=input_shape)

    # Add your EMU block here
    emu_block = EMAU(c=256,k=64, stage_num=3)
    b1_with_emu, _ = emu_block(inputs)

    print(b1_with_emu)

    s1, p1 = encoder_block(b1_with_emu, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net_with_EMU")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    #input_shape = (512, 512, 3)
    model = build_unet_with_emu(input_shape, 9)
    model.summary()