# Using specific libraries
import os
import sys, getopt
import whisper_timestamped as whisper
import matplotlib.pyplot as plt
import gensim.downloader
import re
import numpy as np
from functools import lru_cache
from timeit import repeat
from gensim.models import KeyedVectors

@lru_cache
def vectorize(word, model):
    if word == "[*]":
        return np.zeros(300)
    cleanWord = re.sub('[^A-Za-z0-9]+', '', word)
    try:
        vector = model[cleanWord.lower()]
    except KeyError:
        return np.ones(300)
    return vector

def writeToFile(result, output_file, wv):
    f = open(output_file, 'w')
    f.write("Word,Duration,Confidence\n")

    for i in range(len(result['segments'])):
        for j in range(len(result['segments'][i]['words'])):
            # print(result['segments'][i]['words'][j])
            f.write(f"{result['segments'][i]['words'][j]['text']}" + "," + 
                        f"{str(result['segments'][i]['words'][j]['end'] - result['segments'][i]['words'][j]['start'])}" + "," +
                        f"{result['segments'][i]['words'][j]['confidence']}" + ",")
            word = result['segments'][i]['words'][j]['text']
            if word == "[*]":
                identifier = vectorize(word, wv)
            else: 
                word = re.sub('[^A-Za-z0-9]+', '', word)
                identifier = vectorize(word, wv)
            for j in identifier:
                f.write(str(j) + ",")
            f.write("\n")
    f.close()

def count_words(word_list):
    word_count = {}
    for word in word_list:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def calculate_frequency(result, graph=False):
    words = []
    for i in range(len(result['segments'])):
        for j in range(len(result['segments'][i]['words'])):
            words.append(str(result['segments'][i]['words'][j]['text']).lower())
    counted_words = dict(reversed(sorted(count_words(words).items(), key=lambda x:x[1])))
    
    if graph != False:
        x = list(counted_words.keys())[:10]
        y = list(counted_words.values())[:10]
        
        fig = plt.figure(figsize = (10, 5))
        
        # creating the bar plot
        plt.bar(x, y, color ='maroon',
                width = 0.4)
        
        plt.xlabel("Words")
        plt.ylabel("Num of Occurance")
        plt.title("Frequency of Words")
        plt.show()
    
    return counted_words

def calculate_duration(result, graph=False):
    words = []
    for i in range(len(result['segments'])):
        for j in range(len(result['segments'][i]['words'])):
            words.append(str(result['segments'][i]['words'][j]['text']).lower())
    words = np.asarray(words)
    words = np.unique(words)
    durationOfWords = {}
    for i in words:
        durationOfWords[i] = 0

    for i in range(len(result['segments'])):
        for j in range(len(result['segments'][i]['words'])):
            durationOfWords[str(result['segments'][i]['words'][j]['text']).lower()] += result['segments'][i]['words'][j]['end'] - result['segments'][i]['words'][j]['start']

    durationOfWords = dict(reversed(sorted(durationOfWords.items(), key=lambda x:x[1])))
    if graph != False:
        x = list(durationOfWords.keys())[:10]
        y = list(durationOfWords.values())[:10]
        
        fig = plt.figure(figsize = (10, 5))
        
        # creating the bar plot
        plt.bar(x, y, color ='maroon',
                width = 0.4)
        
        plt.xlabel("Words")
        plt.ylabel("Duration (s)")
        plt.title("Duration of words")
        plt.show()

    return durationOfWords
def main(argv):
    wv = gensim.downloader.load("glove-wiki-gigaword-300")
    # Read command line arguments
    input_file = ''
    output_file = ''
    toGraph = False
    hasAlzheimer = True
    try:
        opts, args = getopt.getopt(argv, "hi:o:g:a:", ["ifile=",'ofile=',"graph=","alzheimer="])
    except getopt.GetoptError:
        print("Transcript_Extraction.py -i <input file (mp3)> -o <output file (csv)>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("Transcript_Extraction.py -i <input file (mp3)> -o <output file (csv)>")
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_file = arg
        elif opt == "-g":
            toGraph = bool(arg)
        elif opt == "-a":
            hasAlzheimer = bool(arg)
    try:
        # Loads in the mp3 file
        audio = whisper.load_audio(str(input_file))
    except RuntimeError:
        if input_file != "" or output_file != "":
            print("No such file or directory")
            sys.exit(2)
    
    # Load whisper models ("small","tiny","medium","large")
    model = whisper.load_model("tiny", device="cpu")

    # Running whisper
    if input_file != "" and output_file != "":
        result = whisper.transcribe(model, audio, language="en", detect_disfluencies=True, plot_word_alignment=toGraph)
        writeToFile(result, output_file, wv)
        # calculate_frequency(result, True)
        calculate_duration(result, True)
    else:
        subfolder_path = ".\\recordings"
        filenames = os.listdir(subfolder_path)
        print(filenames)
        index = 1
        for filename in filenames:
            if hasAlzheimer == True:
                output_file = filename[:4] + "_" + str(index) + "_T.csv"
            else:
                output_file = filename[:4] + "_" + str(index) + "_F.csv"
            audio = ".\\recordings\\"+filename
            result = whisper.transcribe(model, audio, language="en", beam_size=5, best_of=5, detect_disfluencies=True, plot_word_alignment=toGraph)
            writeToFile(result, f".\\transcriptions\\{output_file}", wv)
            index += 1


if __name__ == '__main__':
    main(sys.argv[1:])