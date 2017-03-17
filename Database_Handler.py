#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a helper to connect to a database. 
# It provides several helpful methods.
#
# copyright Paul Baumann
#############################################

import MySQLdb
import warnings

def Get_DB_Handler():
    
#     return Database_Handler.Database_Handler("ADDRESS", 3306, "USERNAME", "PASSWORD", "DBNAME")
    return Database_Handler("127.0.0.1", 3306, "root", "PW4mak!010", "TMC2015")

class Database_Handler:
     
    def __init__(self,host,port,user,user_password,name):
        self._database_host = host
        self._database_port = port
        self._database_user = user
        self._database_user_password = user_password
        self._database_name = name
        
    def printDatabaseInformation(self):
        print "Host: %s" % (self._database_host)
        print "Port: %i" % (self._database_port)
        print "User: %s" % (self._database_user)
        print "User Password: %s" % (self._database_user_password)
        print "Name: %s" % (self._database_name)

    def getNumber(self, number):

        
        if number == 'inf':
            return 50000
        
        if number == '-inf':
            return -50000
        
        return number

    def getString(self, string):
        
        if string != None:        
            newString = string.replace("'","").replace("`","")
            
            if string == 'inf':
                newString = 50000
                
            if string == '-inf':
                newString = -50000
            
            try:
                string_tmp = newString.encode('latin-1','ignore')
                
                return string_tmp
            except UnicodeDecodeError:
                print string
        else:
            return ""
    

    # This method handles database inserts and takes three parameters:
    #
    # table_name: the name of the table in which values should be inserted
    # fields: a list of field names
    # values: a list of values for each field
    #
    # it is to not that the type of each field is determined by the method
    def insert(self, table_name, fields, values):
        
        query = "INSERT INTO %s (" % (table_name)
        
        for i in range(len(fields)-1):
            query = "%s%s," % (query,fields[i])
        
        query = "%s%s) VALUES (" % (query,fields[len(fields)-1])
        
        for i in range(len(values)-1):
            
            if isinstance(values[i],int) or isinstance(values[i],long):
                query = "%s%i," % (query,self.getNumber(values[i]))
            else:
                if isinstance(values[i],str):
                    query = "%s'%s'," % (query,self.getString(values[i]))
                else:
                    if isinstance(values[i],float):
                        query = "%s%f," % (query,self.getNumber(values[i]))
                    else:
                        query = "%s'%s'," % (query,self.getString(values[i]))
        
        i = len(values)-1
        
        if isinstance(values[i],int) or isinstance(values[i],long):
            query = "%s%i)" % (query,self.getNumber(values[i]))
        else:
            if isinstance(values[i],str):
                query = "%s'%s')" % (query,self.getString(values[i]))
            else:
                if isinstance(values[i],float):
                    query = "%s%f)" % (query,self.getNumber(values[i]))
                else:
                    query = "%s'%s')" % (query,self.getString(values[i]))
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        con.autocommit(True)
        cursor = con.cursor()
        id = 0
        try:
            cursor.execute(query)
            id = con.insert_id()         
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0])
            #print query
        
        cursor.close()
        con.close()
        
        return id
        
    def insertMany(self, table_name, fields, values):
        
        values = values.tolist()
        query = "INSERT INTO %s (" % (table_name)
        
        for i in range(len(fields)-1):
            query = "%s%s," % (query,fields[i])
        
        query = "%s%s) VALUES (" % (query,fields[len(fields)-1])
        
        for i in range(len(fields)-1):
            query = "%s%%s," % (query)              
        query = "%s%%s)" % (query)
                
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        id = 0
        try:
            cursor.executemany(query, values)
            id = con.insert_id()
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0])
            #print query
        
        con.autocommit(True)
        cursor.close()
        con.close()
        
        return id
        
    def select(self, query):
        
        result = None
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        
        try:
            cursor.execute(query)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print query
        else:
            result = cursor.fetchall()
        
        cursor.close()
        con.close()
        
        return result
    
    def update(self, query):
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        con.autocommit(True)
        
        try:
            cursor.execute(query)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print query
        
        cursor.close()
        con.close()
        
    
    def dropTable(self, table_name):
        
        query = "DROP TABLE IF EXISTS %s" % (table_name)
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        
        try:
            warnings.filterwarnings("ignore", "Unknown table.*")
            cursor.execute(query)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print query        
        
        cursor.close()
        con.close()
        
    def createTable(self, table_name, selectString):
        
        if selectString == "":
            query = "CREATE TABLE IF NOT EXISTS %s" % (table_name)
        else:
            query = "CREATE TABLE IF NOT EXISTS %s (%s)" % (table_name,selectString)                
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        
        try:
            warnings.filterwarnings("ignore", "Table already exists.*")
            warnings.filterwarnings("ignore", "Unknown table.*")
            cursor.execute(query)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print query        
        
        cursor.close()
        con.close()
        
    def deleteData(self, query):
    
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        
        try:
            warnings.filterwarnings("ignore", "Unknown table.*")
            cursor.execute(query)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print query        
        
        cursor.close()
        con.close()
    
    
    def truncateTable(self, table_name):
        
        query = "TRUNCATE TABLE IF EXISTS %s" % (table_name)
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        
        try:
            warnings.filterwarnings("ignore", "Unknown table.*")
            cursor.execute(query)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print query        
        
        cursor.close()
        con.close()
    
    # Special Methods
    def getGreatestIndex(self, table_name,device_id):
        
        table_id = 0
        
        selectString = "SELECT id FROM %s WHERE device_id = '%s' ORDER BY id DESC LIMIT 1" % (table_name,device_id)
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        
        try:            
            cursor.execute(selectString)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print selectString
        else:
            result = cursor.fetchall()
            
            for row in result:
                table_id = int(row[0])
            
        cursor.close()
        con.close()
        
        return table_id
    
    def getGreatestTimestamp(self, table_name,device_id):
        
        timestamp = 0
        
        selectString = "SELECT timestamp FROM %s WHERE device_id = '%s' ORDER BY id DESC LIMIT 1" % (table_name,device_id)
        
        con = MySQLdb.connect(host=self._database_host, port=self._database_port, user=self._database_user, passwd=self._database_user_password, db=self._database_name)
        cursor = con.cursor()
        
        try:            
            cursor.execute(selectString)            
        except MySQLdb.Error, e:        
            print "Error %d: %s" % (e.args[0], e.args[1])
            print selectString
        else:
            result = cursor.fetchall()
            
            for row in result:
                timestamp = int(row[0])
            
        cursor.close()
        con.close()
        
        return timestamp 
