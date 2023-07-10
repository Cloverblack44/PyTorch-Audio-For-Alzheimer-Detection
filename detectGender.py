from pyAudioAnalysis import audioSegmentation
from pyAudioAnalysis import audioBasicIO
from pydub import AudioSegment

def detect_gender(input_file):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Convert to mono if the audio has multiple channels
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Export the audio as WAV file (required by pyAudioAnalysis)
    temp_wav_file = "temp.wav"
    audio.export(temp_wav_file, format="wav")

    # Perform speaker diarization
    segments = audioSegmentation.speaker_diarization(temp_wav_file, 2)

    # Count the number of segments assigned to each speaker
    num_segments_speaker1 = len(segments[segments == 1])
    num_segments_speaker2 = len(segments[segments == 2])

    # Determine the predominant speaker based on segment count
    if num_segments_speaker1 > num_segments_speaker2:
        predominant_speaker = 1
    else:
        predominant_speaker = 2

    # Perform gender classification on the predominant speaker
    features, _, _ = audioBasicIO.stFeatureExtraction(temp_wav_file, 16000, 8000, 0.050, 0.025)
    gender = audioSegmentation.mtFileClassification(features, "svmSM", "data/svmSM", False)

    # Print the detected gender
    if gender[predominant_speaker - 1] == "male":
        print("Predominant Speaker: Male")
    else:
        print("Predominant Speaker: Female")

    # Delete the temporary WAV file
    os.remove(temp_wav_file)

# Provide the path to the input audio file
input_file_path = "input.wav"

# Call the function to detect the predominant speaker's gender
detect_gender(input_file_path)