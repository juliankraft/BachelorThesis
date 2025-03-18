# MammaliaBox Dataset

## Info
In the 'info' folder all the meta information is stored. Under 'progress_report' there is a chronological log of meetings and data manipulation. In the folder 'labels', there are all the csv files containing labels and information about the files in each session.

## Sessions

The dataset is divided into different sessions. Each session is a collection of recordings from somehow different sources. The following sessions are available:

Session 1: Silvio Aegerter im Rahmen von Wiesel und Co (Herbst 2019)
Session 2: WILMA Summerschool 2020
Session 3: Franz Steffen im Rahmen der Vogelwarte (Frühling 2020)
Session 4: Moritz Nidecker im Rahmen seiner BA (Sommer 2020)
Session 5: Data from Roland Graf (only least weasels, Mustela erminea) -> Images and Videos!
Session 6: Data gathered in Hankensbüttel (only least weasels, Mustela erminea)
Session 7: Data gathered from Nathalie Straub in her Bachelor Thesis (only very little labels available)

### Data structure

For all sessions there is a labels.csv file. The available files are listed - grouped by the sequence they are part of. The following keys are available:

- `SerialNumber`: Serial number of the camera
- `seq_nr`: Sequence number of the recording
- `seq_id`: unique identifier of the sequence (session*10^6 + id)
- `Directory`: Directory of the recording
- `DateTime_start`: Start time of the recording
- `DateTime_end`: End time of the recording
- `duration_seconds`: Duration of the recording in seconds
- `first_file`: First file of the recording
- `last_file`: Last file of the recording
- `n_files`: Number of files in the recording
- `all_files`: All files in the recording
- `label`: Label of the recording
- `duplicate_label`: Indication if there where more than one label per sequence
- `label2`: Simplified and standardized label

Not all sessions have values for all keys available. The following table shows which keys are available for each session:

|   session | SerialNumber   | seq_nr   | seq_id   | Directory   | DateTime_start   | DateTime_end   | duration_seconds   | first_file   | last_file   | n_files   | all_files   | label   | duplicate_label   | label2   |
|----------:|:---------------|:---------|:---------|:------------|:-----------------|:---------------|:-------------------|:-------------|:------------|:----------|:------------|:--------|:------------------|:---------|
|         1 | yes            | yes      | yes      | yes         | yes              | yes            | yes                | yes          | yes         | yes       | yes         | yes     | yes               | yes      |
|         2 | yes            | yes      | yes      | yes         | yes              | yes            | yes                | yes          | yes         | yes       | yes         | yes     | no                | yes      |
|         3 | yes            | yes      | yes      | yes         | yes              | yes            | yes                | yes          | yes         | yes       | yes         | yes     | no                | yes      |
|         4 | yes            | yes      | yes      | yes         | yes              | yes            | yes                | yes          | yes         | yes       | yes         | yes     | yes               | yes      |
|         5 | yes            | yes      | yes      | yes         | yes              | yes            | yes                | yes          | yes         | yes       | yes         | yes     | no                | yes      |
|         6 | yes            | yes      | yes      | yes         | yes              | yes            | yes                | yes          | yes         | yes       | yes         | yes     | no                | yes      |
|         7 | no             | yes      | yes      | yes         | no               | no             | no                 | yes          | yes         | yes       | yes         | yes     | no                | yes      |


## Archive
In the 'archive' folder there are old versions of mutated data stored. They might not be relevant any longer - just in case.
