import serial

arduino = serial.Serial(port='COM9', baudrate=9600)


def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    data = arduino.readline().decode()
    data = int(data) - 1
    return data


while True:
    num = input("Enter a number: ")  # Taking input from user
    value = write_read(num)
    print(value)  # printing the value
