import numpy as np
import pandas
import csv
from operator import attrgetter
class Order:
    customer_id = 0
    order_id = 0
    products = []
    days_since_last_order =0
    dow = 0
    hour = 0
    def toString(self):
        full_string = str(customer_id) + "," + str(order_id) + ","
        counter = 0
        while(counter <134):
            full_string += str(products[counter]) + ","
            counter += 1
        full_string += str(days_since_last_order) + "," + str(dow) + "," + str(hour)
        return full_string
array_of_orders = []





def list_writer(filename, array_of_orders):
    csv = np.genfromtxt(filename, delimiter=",")

    length = len(array_of_orders)
    counter = 0;



    while (counter < (csv.size / csv[0].size)):
        counter2 = 0;
        array_of_orders = array_of_orders + [Order()]
        while (counter2 < csv[counter].size):
            array_of_orders[length+counter].customer_id = csv[counter][counter2]
            counter2 += 1;
            array_of_orders[length+counter].order_id = csv[counter][counter2]
            counter2 += 1
            counter3 = 0
            while (counter3 < 134):
                array_of_orders[length+counter].products += csv[counter][counter2]
                counter2 += 1
                counter3 += 1
            array_of_orders[length+counter].days_since_last_order = csv[counter][counter2]
            counter2 += 1
            array_of_orders[length+counter].dow = csv[counter][counter2]
            counter2 += 1
            array_of_orders[length+counter].hour = csv[counter][counter2]
            counter2 += 1
        counter += 1

    return array_of_orders

counter = 0;
while(counter <=38):

    if(counter == 0):
        filename = "prior000000000000.csv"
    elif(counter < 10):
        filename = "prior00000000000"+str(counter)+".csv"
    else:
        filename = "prior0000000000"+str(counter)+".csv"
    print("Now working on " + filename)
    array_of_orders += list_writer(filename, array_of_orders)
    counter += 1
counter = 0;
while(counter<=3):
    filename = "train00000000000"+str(counter)+".csv"
    print("Now working on " + filename)
    array_of_orders += list_writer(filename,array_of_orders)
    counter += 1
array_of_orders.sort( key = attrgetter('customer_id', 'order_id'))

counter = 0
with open('write.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    counter = 0
    while(counter<len(array_of_orders)):
        writer.writerow(array_of_orders[counter].toString())
        counter += 1
