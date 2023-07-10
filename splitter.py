import os
import speech_recognition as sr

def split_audio_by_speaker(input_file):
    # Create a recognizer instance
    r = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(input_file) as audio_file:
        # Open the audio file as an audio source
        audio = r.record(audio_file)

        # Perform speaker diarization to get speaker segments
        segments = r.separate_speaker_regions(audio)

        # Create a directory to save the segmented audio files
        output_dir = "output_segments"
        os.makedirs(output_dir, exist_ok=True)

        # Process each speaker segment
        for i, segment in enumerate(segments):
            # Save the segment as a separate audio file
            output_file = os.path.join(output_dir, f"segment_{i}.mp3")
            segment.export(output_file, format="wav")

            # Use speech recognition to identify the speaker
            with sr.AudioFile(output_file) as segment_file:
                segment_audio = r.record(segment_file)
                text = r.recognize_google(segment_audio)

            # Print the detected speaker and their corresponding text
            print(f"Segment {i}: Speaker: {text}")

# Provide the path to the input MP3 file
input_file_path = "Baycrest2103.mp3"

# Call the function to split the audio by speaker
split_audio_by_speaker(input_file_path)
