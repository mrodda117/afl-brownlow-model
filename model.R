
### Load required packages and path declaring ###


path <- "insert path here"
setwd(path)
library(dplyr)
library(xgboost)
library(caret)

df <- read.csv(path & "data.csv", stringsAsFactors = FALSE)

### END loading packages ###



#### Match Review Panel Data Wrangling###

df$Reported[is.na(df$Reported)] <- 0
df$Cleared[is.na(df$Cleared)] <- 0
df$Suspended[is.na(df$Suspended)] <- 0

df$Reported[df$Reported == -1] <- NA
df$Cleared[df$Cleared == -1] <- NA
df$Suspended[df$Suspended == -1] <- NA

#If the player was cleared in a review, we subtract that from the total report count
#Reasoning: If they were cleared, the incident might not have been that bad to affect brownlow voting
df$Reported <- df$Reported - df$Cleared
df <- df %>% select(-Cleared)

#Get list of people who are suspended
suspend_list <- df %>% 
  filter(Suspended > 0) %>% 
  select(year, Round, TeamName, playerName)
df$ineligible <- 0

#Once a player has been susepnded, they are ineligigble for the brownlow
#Therefore for every round after they have first been suspended, we mark the player
#as being ineligible
for(i in 1:nrow(suspend_list)){
  df$ineligible[df$Round > suspend_list$Round[i] & 
                  df$playerName == suspend_list$playerName[i] & 
                  df$year == suspend_list$year[i] &
                  df$TeamName == suspend_list$TeamName[i]] <- 1
}
#### END Match Review Panel Data Wrangling ####


#### START Additional feature Creation ####

#Cumuluative Sum of goals through the year
df <- df %>% group_by(year, TeamName, playerName) %>% 
  arrange(Round) %>% 
  mutate(cs_disp = cumsum(disposals), cs_goal = cumsum(goals), cs_report = cumsum(Reported)) %>%
  ungroup()

#Assign the brownlow votes to our prediction value
df$predict_val <- df$votes 

#We standardise disposals to be both a percentage of max disposal for the game, along with rank of disposals within the game
#This should allow the model to work for low disposal games (eg rain/muddy matches that are slow)
df <- df %>% group_by(matchId) %>% 
  mutate(per_disposal = disposals/max(disposals)) %>% 
  mutate(disposal_rank = rank(-disposals)) %>%
  ungroup()

#We do the same standardisation for goals scored in the game - Percentage from Max Goals and Rank of Goals
df <- df %>% group_by(matchId) %>% 
  mutate(per_goal = goals/max(goals)) %>% 
  mutate(goals_rank = rank(-goals)) %>%
  ungroup()

#Disposal Ratio - Ration of handballs to total disposals
df$dis_ratio <- df$handballs/ (df$handballs + df$kicks)
df$dis_ratio[is.na(df$dis_ratio)] <- 0

#Total score that player had in the match (Goals + Behinds)
df$total_score <- df$goals * 6 + df$behinds

#Possesion Ratio - Contested Possessions / Total Possessions
df$cont_ratio <- df$contestedPossessions/(df$contestedPossessions + df$uncontestedPossessions)
df$cont_ratio[is.na(df$cont_ratio)] <- 0

#Mark the players 1 if they were on winning team, 0 if they were not.
df$winning_team <- ifelse(df$margin > 0, 1, 0 )

### END Additional Feature Creation ###


### START Modelling/Prediction ###
#This script was a prototype to practice ensembling
#Its not very effective as its two different versions of gradient boosting trees (High correlation between models)

#Prediction target - Brownlow votes in 2017

#Split the full dataframe into our training set (ie years that arent 2017)
predict_df <- df %>% filter(year != 2017) %>%
  select(-year, -matchId, -Round, -TeamName, -playerName, -votes, -AF, -SC)

#Create xgb matrix from training data
predict_xgb <- xgb.DMatrix(data = as.matrix(select(predict_df, -predict_val)), label = predict_df$predict_val)

#Parameters for xgboost model
#Scale_pos_weight is used due to the highly unbalanced classes (most players dont get votes)
params_list <- list(max_depth = 5,
                    eta = 0.1,
                    subsample = 0.6,
                    colsample_bytree = 0.5,
                    min_child_weight = 1,
                    scale_pos_weight = 10
                    )

#Create a cross validation model
set.seed(117)
cv_model <- xgb.cv(params = params_list,
                   data = predict_xgb,
                   objective = "binary:logistic",
                   nround = 10000,
                   nfold = 10,
                   early_stopping_rounds = 20)

#From our CV Model, use the best iterations and train the actual model used for predictions
xgb_model <- xgb.train(params = params_list,
                       data = predict_xgb,
                       objective = "binary:logistic",
                       nrounds = cv_model$best_iteration)

#Look at the most important features from the model
xgb.importance(feature_names = colnames(as.matrix(select(predict_df, -predict_val))), xgb_model)

#Create a dataframe for actual prediction and use this with the trained xgb model 
test_df <- df %>%
  filter(year == 2017) %>%
  select(-year, -matchId, -Round, -TeamName, -playerName, -votes, -predict_val, -AF, -SC)
probs<- predict(xgb_model, as.matrix(test_df))

joined_df <- df %>% filter(year == 2017) %>% cbind(probs)

#Get the 3 highest predicted probabilities - This is assumed to be the players who got the votes
top3_df <- joined_df %>% 
  group_by(matchId) %>%
  top_n(3, wt = probs) %>% 
  arrange(matchId) %>% 
  mutate(rank = rank(probs))

#Group all the predictions to get the predicted final tally board
final_tally <- top3_df %>% group_by(TeamName, playerName) %>% summarise(votes = sum(rank)) %>% arrange(desc(votes))
#Get the actual votes
actual_tally <- df %>% filter(year==2017) %>% group_by(TeamName, playerName) %>% summarise(act_votes = sum(votes)) %>% arrange(desc(act_votes))
print(final_tally)


#SECOND MODEL - CARET GBM
library(gbm)

#Caret needs out target values as factors (Compared to xgboost which doesnt)
predict_df$predict_val <- predict_df$predict_val %>% as.character()
predict_df$predict_val <- ifelse(predict_df$predict_val == 1, "yes", "no")

#Create the caret variables required for training
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 1,
  classProbs = TRUE)

#Use Carets gridsearch function
gbmGrid <-  expand.grid(interaction.depth = c(3,5,7), 
                        n.trees = (1:5)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = c(10,20,30))

#Create gbm model
gbmFit2 <- train(x = select(predict_df, -predict_val),
                            y = as.factor(predict_df$predict_val),
                            metric = "ROC",
                            method = "gbm", 
                            trControl = fitControl, 
                            verbose = FALSE,
                            tuneGrid = gbmGrid)


#Follow same process as  xgboost model.
#Get training set, predict votes, select top 3 and create an overall tally board.
test_df <- df %>% filter(year == 2017) %>% select(-year, -matchId, -Round, -TeamName, -playerName, -votes, -predict_val, -AF, -SC)
probs2 <- predict(gbmFit2, as.matrix(test_df), type = "prob")$yes

joined_df <- df %>% filter(year == 2017) %>% cbind(probs2)

top3_df <- joined_df %>%
  group_by(matchId) %>% 
  top_n(3, wt = probs2) %>% 
  arrange(matchId) %>% 
  mutate(rank = rank(probs2))

final_tally <- top3_df %>% 
  group_by(TeamName, playerName) %>%
  summarise(votes = sum(rank)) %>% 
  arrange(desc(votes))

actual_tally <- df %>% 
  filter(year==2017) %>%
  group_by(TeamName, playerName) %>%
  summarise(act_votes = sum(votes)) %>% 
  arrange(desc(act_votes))

print(final_tally)



# ENSEMBLE MODEL
# This is a simple xgboost model that uses the 2 probabilites from the xgboost and caret gbm models

# Recreate the probabilities of the models
test_df <- df %>% filter(year != 2017) %>% select(-year, -matchId, -Round, -TeamName, -playerName, -votes, -predict_val, -AF, -SC)
probs2 <- predict(gbmFit2, as.matrix(test_df), type = "prob")$yes

test_df <- df %>% filter(year != 2017) %>% select(-year, -matchId, -Round, -TeamName, -playerName, -votes, -predict_val, -AF, -SC)
probs<- predict(xgb_model, as.matrix(test_df))

#Join up the two probs into a dataframe
model_probs <- data.frame(probs, probs2)
model_probs$predict_val <- ifelse(predict_df$predict_val == "yes", 1, 0)

#Create an xgboost matrix for input into xgb.cv
stage2_xgb <- xgb.DMatrix(data = as.matrix(select(model_probs, -predict_val)), label = model_probs$predict_val)

#Use default parameters
params_list <- list()
set.seed(117)

stage2_cv <- xgb.cv(params = params_list,
                   data = stage2_xgb,
                   metrics = "auc",
                   objective = "binary:logistic",
                   nround = 1000,
                   nfold = 10,
                   early_stopping_rounds = 20)

stage2_model <- xgb.train(params = params_list,
                       data = stage2_xgb,
                       metrics = "auc",
                       objective = "binary:logistic",
                       nrounds = stage2_cv$best_iteration)


#Follow same process - creating prediction set, apply model and get votes. Create a finaly tally board and compare to the actual votes for 2017
test_df <- df %>% filter(year == 2017) %>% select(-year, -matchId, -Round, -TeamName, -playerName, -votes, -predict_val, -AF, -SC)
probs2 <- predict(gbmFit2, as.matrix(test_df), type = "prob")$yes
probs<- predict(xgb_model, as.matrix(test_df))

probs_stage2 <- predict(stage2_model, as.matrix(data.frame(probs, probs2)))

joined_df <- df %>% filter(year == 2017) %>% cbind(probs_stage2)

top3_df <- joined_df %>% 
  group_by(matchId) %>% 
  top_n(3, wt = probs_stage2) %>%
  arrange(matchId) %>%
  mutate(rank = rank(probs_stage2))

final_tally <- top3_df %>%
  group_by(TeamName, playerName) %>%
  summarise(votes = sum(rank)) %>%
  arrange(desc(votes))

actual_tally <- df %>%
  filter(year==2017) %>%
  group_by(TeamName, playerName) %>%
  summarise(act_votes = sum(votes)) %>%
  arrange(desc(act_votes))

print(final_tally)




