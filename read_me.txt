Run the scripts below from cs229_project. 

1. The first step is to run the train pipeline. We run the pipeline using bash scripts. Before running the script, go to scripts/settings.sh, and see if the directories are correct. The python file will be using directories provided here to export the model output. 

2. Enter the following command in the terminal:
   
   bash scripts/run_eng.sh 
   bash scripts/run_ch.sh 

Note that the working directory is cs229_project. Do not cd into scripts. 

The two scripts runs the training pipeline for English and Chinese datasets. Since this is the training part, it might take a while for the run to complete. The outputs are saved to the directory output. If training is done for multiple values of K (number of topics), the output is saved in folder called train_all_{"language"}_{"type of document"}. This folder is inside the folder output. If the model is run for specific K, the output is saved to folder: train_k_{"language"}_{"type of document"}

3. Next run the following command:

    bash scripts/run_tester.sh 

This looks at the trained model and computes perplexity and coherence. It also generates figures (saved to folder called figures). We can use the figure for preliminary analysis. 

4. Enter the following command:
    
    python generate_training_figs.py 

This automatically generates the plots for training, we will be using in our report. 

5. Run 

  bash scripts/run_word_analyzer.sh 

This generates the top words found by each model and exports the words as a civ file. Use this for qualitative analysis. 

6. Finally run 

    python correlation_analyzer.py -l english 
    python correlation_analyzer.py -l chinese 


