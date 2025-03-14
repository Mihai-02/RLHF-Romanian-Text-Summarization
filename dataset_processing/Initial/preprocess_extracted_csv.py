import tensorflow as tf

import numpy as np
import pandas as pd

import re

from sklearn.model_selection import train_test_split


chars_to_replace_lower = ["[áàãảȧạäåḁāąᶏⱥȀȁấầẫẩậắằẵẳặǻǡǟȃɑⱯɐɒａæᵆǽǣᴂ]",
"[ḃḅḇƀɓƃᶀｂȸ]",
"[ćĉčċc̄çḉȼƈɕᴄｃ]",
"[ďḋḑd̦ḍḓḏđðd̦ɖɗƌᶁȡｄÞþȸǱǲǳždžǅǆ]",
"[éèêḙěĕẽḛẻėëēȩęᶒɇȅếềễểḝḗḕȇẹệⱸƏəƎǝƐɛｅᴂᴔÆæᴁᴭᵆǼǽǢǣŒœᵫ]",
"[ḟƒᵮᶂꜰＦｆﬀﬃﬄﬁﬂ]",
"[ǵğĝǧġģḡǥɠᶃɢȜȝｇŋɢɢ̆]",
"[ĥȟḧḣḩḥḫẖħⱨɦʰｈh̃ɧ]",
"[íìĭǐïḯĩįīỉȉịḭɨᶖiıƖɩｉﬁĲĳ]",
"[ĵɉǰȷɟʄᴊｊĳǈǉǋǌʲj̃]",
"[ḱǩķḲḳḴḵƘƙⱩᶄᶄꝁᴋｋ]",
"[ĺľļḷḹḽḻłŀƚⱡɫᶅɭȴʟｌﬂǇǈǉ]",
"[ḿṁṃᵯᶆɱᴍｍ]",
"[ńǹňñṅņṇṋṉn̈ɲƞᵰɳȵɴｎŋǋǌ]",
"[óòŏôốồỗổǑǒöȫőõṍṏȭȯȱøǿǫǭōṓṑỏȍȏơớờỡởợọộƟɵƆɔȢȣⱺᴏｏᴔ]",
"[ṕṗᵽƥp̃ᶈǷｐȹ]",
"[ɋƣʠｑȹ]",
"[ŕřṙŗȑȓṛṝṟɍɽꝛᶉɼɾᵳʀｒɹɹʁ]",
"[ẞśṥŝšṧṡᵴᶊʂȿꜱƩʃｓ]",
"[ťṫṱṯŧⱦƭʈẗᵵƫȶᶙｔ]",
"[úùŭûǔůüǘǜǚǖűũṹųūṻủȕȗưứừữửựụṳṷṵʉʊȢᵾᶙᴜｕᵫɯ]",
"[ṽṿʋᶌⱱᴠʌｖ]",
"[ẃẁŵẅẇẉẘⱳᴡｗʍw̃]",
"[ẍẋᶍｘ]",
"[ýỳŷẙÿỹẏȳỷỵɎɏƴʏｙ]",
"[źẑžżẓẕƶȥⱬᵶᶎʐʑɀᴢƷʒƸƹｚǲǳžžǅǆ]"]

chars_to_replace_upper = ['[ÁÀÃẢȦẠÄÅḀĀĄȺẤẦẪẨẬẮẰẴẲẶǺǠǞȂⱭᴀＡÆᴁᴭǼǢ]',
 '[ḂḄḆɃƁƂᵬʙＢ]',
 '[ĆĈČĊCÇḈȻƇＣ]',
 '[ĎḊḐDḌḒḎĐÐDƉƊƋᵭᶑᴅＤDŽDǄ]',
 '[ÉÈÊḘĚĔẼḚẺĖËĒȨĘɆȄẾỀỄỂḜḖḔȆẸỆᴇＥ]',
 '[ḞƑ]',
 '[ǴĞĜǦĠĢḠǤƓＧŊ]',
 '[ĤȞḦḢḨḤḪH̱ĦⱧʜＨ]',
 '[ÍÌĬǏÏḮĨĮĪỈȈỊḬƗᵻİIɪＩ]',
 '[ĴɈJ̌ʝＪĲǇǊ]',
 '[ḰǨĶⱪꝀＫ]',
 '[ĹĽĻḶḸḼḺŁĿȽⱠⱢɬＬ]',
 '[ḾṀṂⱮＭ]',
 '[ŃǸŇÑṄŅṆṊṈNƝȠᶇＮŊǊ]',
 '[ÓÒŎÔỐỒỖỔÖȪŐÕṌṎȬȮȰØǾǪǬŌṒṐỎȌȎƠỚỜỠỞỢỌỘ]',
 '[ṔṖⱣƤPᵱᴘƿＰ]',
 '[ɊƢＱ]',
 '[ŔŘṘŖȐȒṚṜṞɌⱤꝚᵲＲ]',
 '[ſßŚṤŜŠṦṠẛṨṩＳ]',
 '[ŤṪṰṮŦȾƬƮT̈ᴛＴ]',
 '[ÚÙŬÛǓŮÜǗǛǙǕŰŨṸŲŪṺỦȔȖƯỨỪỮỬỰỤṲṶṴɄƱȣＵ]',
 '[ṼṾƲⱴɅＶ]',
 '[ẂẀŴẄẆẈW̊ⱲＷʷ]',
 '[ẌẊＸ]',
 '[ÝỲŶŸỸẎȲỶỴƳＹ]',
 '[ŹẐŽŻẒẔƵȤⱫＺŽ]']

alphabet_lower = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
           's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
alphabet_upper = [x.upper() for x in alphabet_lower]


dataset = pd.read_csv("/kaggle/input/set-updated-1/(FINAL)set_updated_deleted_summary_maiMareDecat_content.csv")



#Republica Moldova la Jocurile Olimpice de iarnă din 2002
#Premiul Oscar pentru cele mai bune decoruri
#Alegeri pentru Parlamentul European, 2004   ??
#Cupa Campionilor Europeni 1963-1964 ??#
#UTC-5
#1996 în film
#Sezonul de Formula 1 din 2012
#2012 în astronomie
#FC Dinamo in sezonul...
#Divizia B 2005-2006... / Serie A ...
#1902 în literatură
#Cupa UEFA
#Serie A ...
#Octombrie 1990
#Glosar de cinematografie
#Premiul ...
#Brigada 22 Infanterie (1916-1918)
#Regimentul 5 Vânători (1916-1918)
#Plasa Lipova, județul Timiș-Torontal
#Preliminariile Campionatului European de Fotbal

#patterns = []   #TOATE DE MAI SUS + CELELALTE
#for patternDel in patterns:
#    to_delete= dataset['Title'].str.contains(patternDel)

patternDel = r'Preliminariile '
to_delete = dataset['Title'].str.contains(patternDel)

dataset = dataset[~to_delete]

#dataset.to_csv("set_final.csv")

#STERGERE ARTICOLE DUPA LUNGIME SUMAR
dataset.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
dataset['summary_word_count'] = dataset['Summary'].apply(lambda x: len(x.split()))
dataset['content_word_count'] = dataset['Content'].apply(lambda x: len(x.split()))

dataset = dataset[dataset["summary_word_count"]<dataset["content_word_count"]]

#temp[temp["summary_word_count"]*2<=temp["content_word_count"]]

dataset.to_csv("set_final.csv")

