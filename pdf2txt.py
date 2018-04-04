# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:26:50 2017

@author: Moukthika
"""

import PyPDF2, os
#loading data
i = 0
indir = "path to  folder containing pdfs"
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        p = open("/Users/Moukthika/Desktop/pdf_extract/"+str(i)+".txt",'w', encoding = 'utf-8')
        i += 1
        try:
            #log = open(os.path.join(root, f), 'rb')
            pdfFileObj = open(os.path.join(root, f), 'rb')
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            pdfReader.numPages
            print(f)
            for pageNum in range(0, pdfReader.numPages):   #extracting text from all the pdfs at a time
                pageObj = pdfReader.getPage(pageNum)
                pdf_content = pageObj.extractText()
                print(pdf_content)
                p.write(pdf_content)   
        
        except:
            print("PDF_error")
            pdfFileObj.close()
