import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from pandas import DataFrame
import csv
from parsel import Selector
import random
import re

def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
        return True
    except NoSuchElementException:
        print("erreur")
        return False
def check_exists_by_class_name(name):
    try:
        driver.find_element_by_class_name(name)
        return True
    except NoSuchElementException:
        print("erreur")
        return False
k=0
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get('https://www.linkedin.com')
time.sleep(0.5)
driver.find_element_by_xpath('/html/body/div[1]/div/section/div/div[2]/button[2]').click()
time.sleep(0.5)
username = driver.find_element_by_id('session_key')
username.send_keys("erhRE.HERHrt@gmail.com")
time.sleep(0.5)
password = driver.find_element_by_id('session_password')
password.send_keys("testscrap")
time.sleep(0.5)
sign_in_button = driver.find_element_by_xpath('//*[@type="submit"]')
sign_in_button.click()
driver.get('https:www.google.com')
time.sleep(3)
driver.find_element_by_xpath('/html/body/div[2]/div[2]/div[3]/span/div/div/div/div[3]/button[2]/div').click()
search_query = driver.find_element_by_name('q')
search_query.send_keys('site:linkedin.com/in/ AND "python" AND "London"')
time.sleep(0.5)
search_query.send_keys(Keys.RETURN)
all_url=[]
for u in range(32):
    elems = driver.find_elements_by_xpath("//a[@href]")
    for elem in elems:
        if "uk.linkedin.com" in elem.get_attribute("href") and "google" not in elem.get_attribute("href") and "translate" not in elem.get_attribute("href"):
            all_url.append(elem.get_attribute("href"))
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    temps=random.randint(2, 5)
    time.sleep(temps)
    while check_exists_by_xpath("/html/body/div[7]/div/div[10]/div[1]/div/div[6]/span[1]/table/tbody/tr/td[12]/a") == False:
        time.sleep(temps)
    time.sleep(temps)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.find_element_by_xpath("/html/body/div[7]/div/div[10]/div[1]/div/div[6]/span[1]/table/tbody/tr/td[12]/a").click()    
for linkedin_url in all_url:

   # get the profile URL
   driver.get(linkedin_url)

   # add a 5 second pause loading each URL
   time.sleep(5)

   # assigning the source code for the webpage to variable sel
   sel = Selector(text=driver.page_source)
   # xpath to extract the text from the class containing the name
   name = sel.xpath('//*[starts-with(@class, "text-heading-xlarge inline t-24 v-align-middle break-words")]/text()').extract_first()
   
   if name:
       name = name.strip()
   # xpath to extract the text from the class containing the job title
   job_title = sel.xpath('//*[starts-with(@class, "text-body-medium break-words")]/text()').extract_first()
   if job_title:
       job_title = job_title.strip()
   
   # xpath to extract the text from the class containing the company
   Y=0
   last_height = driver.execute_script("return window.pageYOffset")
   while (check_exists_by_xpath("/html/body/div[7]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section/div[2]/button") == False and check_exists_by_xpath("/html/body/div[6]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section/div[2]/button") == False):
          Y+=500
          driver.execute_script("window.scrollTo(0, "+str(Y)+")")
          new_height = driver.execute_script("return window.pageYOffset")
          time.sleep(1)
          if new_height == last_height:
             break
          last_height = new_height        

   if check_exists_by_xpath("/html/body/div[7]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section/div[2]/button") ==True:
       driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section/div[2]/button").click()
       description= driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section").text
       description =description.split('\n')
       i=0
       k+=1
       while i <len(description):
            if "See" in description[i]:
                del description[i]
                continue
            if description[i].isdigit():
                del description[i]
                continue
            if "Skills" in description[i] or "skill" in description[i] or "Skill" in description[i] or "skills" in description[i]:
                del description[i]
                continue
            if "Endorsed" in description[i]:
                del description[i]
                continue
            if "Show less" in description[i]:
                del description[i]
                continue
            i+=1
       C = {'id':k,'name': [name],'job_title': [job_title],'competence': [description], 'contact': linkedin_url}
       donnees = DataFrame(C, columns= ['id','name', 'job_title', 'competence','contact'])
       export_csv = donnees.to_csv ("candidat4.csv", mode='a',index = None, header=False, encoding='utf8', sep=';')
   if check_exists_by_xpath("/html/body/div[6]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section/div[2]/button") ==True:
       driver.find_element_by_xpath("/html/body/div[6]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section/div[2]/button").click()
       description= driver.find_element_by_xpath("/html/body/div[6]/div[3]/div/div/div/div/div[3]/div/div/main/div/div/div[6]/div/section").text
       description =description.split('\n')
       i=0
       k+=1
       while i <len(description):
            if "See" in description[i]:
                del description[i]
                continue
            if description[i].isdigit():
                del description[i]
                continue
            if "Skills" in description[i] or "skill" in description[i] or "Skill" in description[i] or "skills" in description[i]:
                del description[i]
                continue
            if "Endorsed" in description[i]:
                del description[i]
                continue
            if "Show less" in description[i]:
                del description[i]
                continue
            i+=1
       C = {'id':k,'name': [name],'job_title': [job_title],'competence': [description], 'contact': linkedin_url}
       donnees = DataFrame(C, columns= ['id','name', 'job_title', 'competence','contact'])
       export_csv = donnees.to_csv ("candidat4.csv", mode='a',index = None, header=False, encoding='utf8', sep=';')


linkedin_url = driver.current_url
driver.quit()