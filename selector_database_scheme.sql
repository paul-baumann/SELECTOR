# ************************************************************
# Sequel Pro SQL dump
# Version 4135
#
# http://www.sequelpro.com/
# http://code.google.com/p/sequel-pro/
#
# Host: 127.0.0.1 (MySQL 5.5.9)
# Database: PaperUbiComp2014
# Generation Time: 2016-09-29 12:04:55 +0000
# ************************************************************


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;


# Dump of table NextPlace_Feature_Matrix
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextPlace_Feature_Matrix`;

CREATE TABLE `NextPlace_Feature_Matrix` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `userID` int(11) DEFAULT NULL,
  `timestamp` bigint(20) DEFAULT NULL,
  `date` varchar(11) DEFAULT NULL,
  `time` varchar(11) DEFAULT NULL,
  `no_features` int(11) DEFAULT NULL,
  `contains_nans` tinyint(1) DEFAULT NULL,
  `ground_truth` int(11) DEFAULT NULL,
  `NI1` double DEFAULT NULL,
  `NI2` double DEFAULT NULL,
  `NI3` double DEFAULT NULL,
  `NI4` double DEFAULT NULL,
  `NI5` double DEFAULT NULL,
  `NI6` double DEFAULT NULL,
  `TI1` int(11) DEFAULT NULL,
  `TI2` double DEFAULT NULL,
  `TI3` double DEFAULT NULL,
  `TI4` double DEFAULT NULL,
  `TI5` double DEFAULT NULL,
  `TI6` double DEFAULT NULL,
  `TI7` double DEFAULT NULL,
  `TI8` double DEFAULT NULL,
  `TI9` double DEFAULT NULL,
  `TI10` double DEFAULT NULL,
  `TI11` double DEFAULT NULL,
  `TI12` double DEFAULT NULL,
  `TI13` double DEFAULT NULL,
  `TI14` double DEFAULT NULL,
  `TI15` double DEFAULT NULL,
  `SI1` double DEFAULT NULL,
  `SI2` double DEFAULT NULL,
  `SI3` double DEFAULT NULL,
  `SI4` double DEFAULT NULL,
  `SI5` double DEFAULT NULL,
  `UI1` double DEFAULT NULL,
  `UI2` double DEFAULT NULL,
  `UI3` double DEFAULT NULL,
  `UI4` double DEFAULT NULL,
  `UI5` double DEFAULT NULL,
  `UI6` double DEFAULT NULL,
  `UI7` double DEFAULT NULL,
  `UI8` float DEFAULT NULL,
  `UI9` double DEFAULT NULL,
  `UI10` double DEFAULT NULL,
  `UI11` double DEFAULT NULL,
  `UI12` double DEFAULT NULL,
  `UI13` double DEFAULT NULL,
  `UI14` double DEFAULT NULL,
  `UI15` double DEFAULT NULL,
  `UI16` double DEFAULT NULL,
  `UI17` double DEFAULT NULL,
  `UI18` double DEFAULT NULL,
  `UI19` double DEFAULT NULL,
  `UI20` double DEFAULT NULL,
  `UI21` double DEFAULT NULL,
  `UI22` double DEFAULT NULL,
  `UI23` double DEFAULT NULL,
  `UI24` bigint(20) DEFAULT NULL,
  `UI25` double DEFAULT NULL,
  `UI26` double DEFAULT NULL,
  `UI27` double DEFAULT NULL,
  `UI28` double DEFAULT NULL,
  PRIMARY KEY (`ID`),
  KEY `userID` (`userID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextPlace_Pre_Selected_Features
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextPlace_Pre_Selected_Features`;

CREATE TABLE `NextPlace_Pre_Selected_Features` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `userID` int(11) DEFAULT NULL,
  `NI1` tinyint(1) DEFAULT NULL,
  `NI2` tinyint(1) DEFAULT NULL,
  `NI3` tinyint(1) DEFAULT NULL,
  `NI4` tinyint(1) DEFAULT NULL,
  `NI5` tinyint(1) DEFAULT NULL,
  `NI6` tinyint(1) DEFAULT NULL,
  `TI1` tinyint(1) DEFAULT NULL,
  `TI2` tinyint(1) DEFAULT NULL,
  `TI3` tinyint(1) DEFAULT NULL,
  `TI4` tinyint(1) DEFAULT NULL,
  `TI5` tinyint(1) DEFAULT NULL,
  `TI6` tinyint(1) DEFAULT NULL,
  `TI7` tinyint(1) DEFAULT NULL,
  `TI8` tinyint(1) DEFAULT NULL,
  `TI9` tinyint(1) DEFAULT NULL,
  `TI10` tinyint(1) DEFAULT NULL,
  `TI11` tinyint(1) DEFAULT NULL,
  `TI12` tinyint(1) DEFAULT NULL,
  `TI13` tinyint(1) DEFAULT NULL,
  `TI14` tinyint(1) DEFAULT NULL,
  `TI15` tinyint(1) DEFAULT NULL,
  `SI1` tinyint(1) DEFAULT NULL,
  `SI2` tinyint(1) DEFAULT NULL,
  `SI3` tinyint(1) DEFAULT NULL,
  `SI4` tinyint(1) DEFAULT NULL,
  `SI5` tinyint(1) DEFAULT NULL,
  `UI1` tinyint(1) DEFAULT NULL,
  `UI2` tinyint(1) DEFAULT NULL,
  `UI3` tinyint(1) DEFAULT NULL,
  `UI4` tinyint(1) DEFAULT NULL,
  `UI5` tinyint(1) DEFAULT NULL,
  `UI6` tinyint(1) DEFAULT NULL,
  `UI7` tinyint(1) DEFAULT NULL,
  `UI8` tinyint(1) DEFAULT NULL,
  `UI9` tinyint(1) DEFAULT NULL,
  `UI10` tinyint(1) DEFAULT NULL,
  `UI11` tinyint(1) DEFAULT NULL,
  `UI12` tinyint(1) DEFAULT NULL,
  `UI13` tinyint(1) DEFAULT NULL,
  `UI14` tinyint(1) DEFAULT NULL,
  `UI15` tinyint(1) DEFAULT NULL,
  `UI16` tinyint(1) DEFAULT NULL,
  `UI17` tinyint(1) DEFAULT NULL,
  `UI18` tinyint(1) DEFAULT NULL,
  `UI19` tinyint(1) DEFAULT NULL,
  `UI20` tinyint(1) DEFAULT NULL,
  `UI21` tinyint(1) DEFAULT NULL,
  `UI22` tinyint(1) DEFAULT NULL,
  `UI23` tinyint(1) DEFAULT NULL,
  `UI24` tinyint(1) DEFAULT NULL,
  `UI25` tinyint(1) DEFAULT NULL,
  `UI26` tinyint(1) DEFAULT NULL,
  `UI27` tinyint(1) DEFAULT NULL,
  `UI28` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`ID`),
  KEY `userID` (`userID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextPlace_Prediction_Result_Analysis
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextPlace_Prediction_Result_Analysis`;

CREATE TABLE `NextPlace_Prediction_Result_Analysis` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `run_id` int(11) DEFAULT NULL,
  `feature_combination` varchar(255) DEFAULT NULL,
  `final` tinyint(1) DEFAULT NULL,
  `accuracy` double DEFAULT NULL,
  `precis` double DEFAULT NULL,
  `recall` double DEFAULT NULL,
  `fscore` double DEFAULT NULL,
  `kappa_random` double DEFAULT NULL,
  `kappa_histogram` double DEFAULT NULL,
  `kappa_dominating` double DEFAULT NULL,
  `mcc` double DEFAULT NULL,
  `frequency_of_top_class` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `run_id` (`run_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextPlace_Prediction_Run
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextPlace_Prediction_Run`;

CREATE TABLE `NextPlace_Prediction_Run` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` int(11) DEFAULT NULL,
  `start_timestamp` varchar(30) DEFAULT NULL,
  `end_timestamp` varchar(30) DEFAULT NULL,
  `selected_algorithm` varchar(30) DEFAULT NULL,
  `selected_metric` varchar(30) DEFAULT NULL,
  `number_of_optimization_data` int(11) DEFAULT NULL,
  `number_of_training_data` int(11) DEFAULT NULL,
  `number_of_test_data` int(11) DEFAULT NULL,
  `number_of_total_data` int(11) DEFAULT NULL,
  `optimization_array` longtext,
  `training_array` longtext,
  `test_array` longtext,
  `is_network` tinyint(1) DEFAULT NULL,
  `is_temporal` tinyint(1) DEFAULT NULL,
  `is_spatial` tinyint(1) DEFAULT NULL,
  `is_context` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotPlace_Feature_Matrix
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotPlace_Feature_Matrix`;

CREATE TABLE `NextSlotPlace_Feature_Matrix` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `userID` int(11) DEFAULT NULL,
  `timestamp` bigint(20) DEFAULT NULL,
  `date` varchar(11) DEFAULT NULL,
  `time` varchar(11) DEFAULT NULL,
  `no_features` int(11) DEFAULT NULL,
  `contains_nans` tinyint(1) DEFAULT NULL,
  `ground_truth` int(11) DEFAULT NULL,
  `NI1` double DEFAULT NULL,
  `NI2` double DEFAULT NULL,
  `NI3` double DEFAULT NULL,
  `NI4` double DEFAULT NULL,
  `NI5` double DEFAULT NULL,
  `NI6` double DEFAULT NULL,
  `TI1` int(11) DEFAULT NULL,
  `TI2` double DEFAULT NULL,
  `TI3` double DEFAULT NULL,
  `TI4` double DEFAULT NULL,
  `TI5` double DEFAULT NULL,
  `TI6` double DEFAULT NULL,
  `TI7` double DEFAULT NULL,
  `TI8` double DEFAULT NULL,
  `TI9` double DEFAULT NULL,
  `TI10` double DEFAULT NULL,
  `TI11` double DEFAULT NULL,
  `TI12` double DEFAULT NULL,
  `TI13` double DEFAULT NULL,
  `TI14` double DEFAULT NULL,
  `TI15` double DEFAULT NULL,
  `SI1` double DEFAULT NULL,
  `SI2` double DEFAULT NULL,
  `SI3` double DEFAULT NULL,
  `SI4` double DEFAULT NULL,
  `SI5` double DEFAULT NULL,
  `UI1` double DEFAULT NULL,
  `UI2` double DEFAULT NULL,
  `UI3` double DEFAULT NULL,
  `UI4` double DEFAULT NULL,
  `UI5` double DEFAULT NULL,
  `UI6` double DEFAULT NULL,
  `UI7` double DEFAULT NULL,
  `UI8` float DEFAULT NULL,
  `UI9` double DEFAULT NULL,
  `UI10` double DEFAULT NULL,
  `UI11` double DEFAULT NULL,
  `UI12` double DEFAULT NULL,
  `UI13` double DEFAULT NULL,
  `UI14` double DEFAULT NULL,
  `UI15` double DEFAULT NULL,
  `UI16` double DEFAULT NULL,
  `UI17` double DEFAULT NULL,
  `UI18` double DEFAULT NULL,
  `UI19` double DEFAULT NULL,
  `UI20` double DEFAULT NULL,
  `UI21` double DEFAULT NULL,
  `UI22` double DEFAULT NULL,
  `UI23` double DEFAULT NULL,
  `UI24` bigint(20) DEFAULT NULL,
  `UI25` double DEFAULT NULL,
  `UI26` double DEFAULT NULL,
  `UI27` double DEFAULT NULL,
  `UI28` double DEFAULT NULL,
  PRIMARY KEY (`ID`),
  KEY `userID` (`userID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotPlace_Pre_Selected_Features
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotPlace_Pre_Selected_Features`;

CREATE TABLE `NextSlotPlace_Pre_Selected_Features` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `userID` int(11) DEFAULT NULL,
  `NI1` tinyint(1) DEFAULT NULL,
  `NI2` tinyint(1) DEFAULT NULL,
  `NI3` tinyint(1) DEFAULT NULL,
  `NI4` tinyint(1) DEFAULT NULL,
  `NI5` tinyint(1) DEFAULT NULL,
  `NI6` tinyint(1) DEFAULT NULL,
  `TI1` tinyint(1) DEFAULT NULL,
  `TI2` tinyint(1) DEFAULT NULL,
  `TI3` tinyint(1) DEFAULT NULL,
  `TI4` tinyint(1) DEFAULT NULL,
  `TI5` tinyint(1) DEFAULT NULL,
  `TI6` tinyint(1) DEFAULT NULL,
  `TI7` tinyint(1) DEFAULT NULL,
  `TI8` tinyint(1) DEFAULT NULL,
  `TI9` tinyint(1) DEFAULT NULL,
  `TI10` tinyint(1) DEFAULT NULL,
  `TI11` tinyint(1) DEFAULT NULL,
  `TI12` tinyint(1) DEFAULT NULL,
  `TI13` tinyint(1) DEFAULT NULL,
  `TI14` tinyint(1) DEFAULT NULL,
  `TI15` tinyint(1) DEFAULT NULL,
  `SI1` tinyint(1) DEFAULT NULL,
  `SI2` tinyint(1) DEFAULT NULL,
  `SI3` tinyint(1) DEFAULT NULL,
  `SI4` tinyint(1) DEFAULT NULL,
  `SI5` tinyint(1) DEFAULT NULL,
  `UI1` tinyint(1) DEFAULT NULL,
  `UI2` tinyint(1) DEFAULT NULL,
  `UI3` tinyint(1) DEFAULT NULL,
  `UI4` tinyint(1) DEFAULT NULL,
  `UI5` tinyint(1) DEFAULT NULL,
  `UI6` tinyint(1) DEFAULT NULL,
  `UI7` tinyint(1) DEFAULT NULL,
  `UI8` tinyint(1) DEFAULT NULL,
  `UI9` tinyint(1) DEFAULT NULL,
  `UI10` tinyint(1) DEFAULT NULL,
  `UI11` tinyint(1) DEFAULT NULL,
  `UI12` tinyint(1) DEFAULT NULL,
  `UI13` tinyint(1) DEFAULT NULL,
  `UI14` tinyint(1) DEFAULT NULL,
  `UI15` tinyint(1) DEFAULT NULL,
  `UI16` tinyint(1) DEFAULT NULL,
  `UI17` tinyint(1) DEFAULT NULL,
  `UI18` tinyint(1) DEFAULT NULL,
  `UI19` tinyint(1) DEFAULT NULL,
  `UI20` tinyint(1) DEFAULT NULL,
  `UI21` tinyint(1) DEFAULT NULL,
  `UI22` tinyint(1) DEFAULT NULL,
  `UI23` tinyint(1) DEFAULT NULL,
  `UI24` tinyint(1) DEFAULT NULL,
  `UI25` tinyint(1) DEFAULT NULL,
  `UI26` tinyint(1) DEFAULT NULL,
  `UI27` tinyint(1) DEFAULT NULL,
  `UI28` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`ID`),
  KEY `userID` (`userID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotPlace_Prediction_Result_Analysis
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotPlace_Prediction_Result_Analysis`;

CREATE TABLE `NextSlotPlace_Prediction_Result_Analysis` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `run_id` int(11) DEFAULT NULL,
  `feature_combination` varchar(255) DEFAULT NULL,
  `final` tinyint(1) DEFAULT NULL,
  `accuracy` double DEFAULT NULL,
  `precis` double DEFAULT NULL,
  `recall` double DEFAULT NULL,
  `fscore` double DEFAULT NULL,
  `kappa_random` double DEFAULT NULL,
  `kappa_histogram` double DEFAULT NULL,
  `kappa_dominating` double DEFAULT NULL,
  `mcc` double DEFAULT NULL,
  `frequency_of_top_class` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `run_id` (`run_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotPlace_Prediction_Run
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotPlace_Prediction_Run`;

CREATE TABLE `NextSlotPlace_Prediction_Run` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` int(11) DEFAULT NULL,
  `start_timestamp` varchar(30) DEFAULT NULL,
  `end_timestamp` varchar(30) DEFAULT NULL,
  `selected_algorithm` varchar(30) DEFAULT NULL,
  `selected_metric` varchar(30) DEFAULT NULL,
  `number_of_optimization_data` int(11) DEFAULT NULL,
  `number_of_training_data` int(11) DEFAULT NULL,
  `number_of_test_data` int(11) DEFAULT NULL,
  `number_of_total_data` int(11) DEFAULT NULL,
  `optimization_array` longtext,
  `training_array` longtext,
  `test_array` longtext,
  `is_network` tinyint(1) DEFAULT NULL,
  `is_temporal` tinyint(1) DEFAULT NULL,
  `is_spatial` tinyint(1) DEFAULT NULL,
  `is_context` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotTransition_Feature_Matrix
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotTransition_Feature_Matrix`;

CREATE TABLE `NextSlotTransition_Feature_Matrix` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `userID` int(11) DEFAULT NULL,
  `timestamp` bigint(20) DEFAULT NULL,
  `date` varchar(11) DEFAULT NULL,
  `time` varchar(11) DEFAULT NULL,
  `no_features` int(11) DEFAULT NULL,
  `contains_nans` tinyint(1) DEFAULT NULL,
  `ground_truth` int(11) DEFAULT NULL,
  `NI1` double DEFAULT NULL,
  `NI2` double DEFAULT NULL,
  `NI3` double DEFAULT NULL,
  `NI4` double DEFAULT NULL,
  `NI5` double DEFAULT NULL,
  `NI6` double DEFAULT NULL,
  `TI1` int(11) DEFAULT NULL,
  `TI2` double DEFAULT NULL,
  `TI3` double DEFAULT NULL,
  `TI4` double DEFAULT NULL,
  `TI5` double DEFAULT NULL,
  `TI6` double DEFAULT NULL,
  `TI7` double DEFAULT NULL,
  `TI8` double DEFAULT NULL,
  `TI9` double DEFAULT NULL,
  `TI10` double DEFAULT NULL,
  `TI11` double DEFAULT NULL,
  `TI12` double DEFAULT NULL,
  `TI13` double DEFAULT NULL,
  `TI14` double DEFAULT NULL,
  `TI15` double DEFAULT NULL,
  `SI1` double DEFAULT NULL,
  `SI2` double DEFAULT NULL,
  `SI3` double DEFAULT NULL,
  `SI4` double DEFAULT NULL,
  `SI5` double DEFAULT NULL,
  `UI1` double DEFAULT NULL,
  `UI2` double DEFAULT NULL,
  `UI3` double DEFAULT NULL,
  `UI4` double DEFAULT NULL,
  `UI5` double DEFAULT NULL,
  `UI6` double DEFAULT NULL,
  `UI7` double DEFAULT NULL,
  `UI8` float DEFAULT NULL,
  `UI9` double DEFAULT NULL,
  `UI10` double DEFAULT NULL,
  `UI11` double DEFAULT NULL,
  `UI12` double DEFAULT NULL,
  `UI13` double DEFAULT NULL,
  `UI14` double DEFAULT NULL,
  `UI15` double DEFAULT NULL,
  `UI16` double DEFAULT NULL,
  `UI17` double DEFAULT NULL,
  `UI18` double DEFAULT NULL,
  `UI19` double DEFAULT NULL,
  `UI20` double DEFAULT NULL,
  `UI21` double DEFAULT NULL,
  `UI22` double DEFAULT NULL,
  `UI23` double DEFAULT NULL,
  `UI24` bigint(20) DEFAULT NULL,
  `UI25` double DEFAULT NULL,
  `UI26` double DEFAULT NULL,
  `UI27` double DEFAULT NULL,
  `UI28` double DEFAULT NULL,
  PRIMARY KEY (`ID`),
  KEY `userID` (`userID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotTransition_Pre_Selected_Features
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotTransition_Pre_Selected_Features`;

CREATE TABLE `NextSlotTransition_Pre_Selected_Features` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `userID` int(11) DEFAULT NULL,
  `NI1` tinyint(1) DEFAULT NULL,
  `NI2` tinyint(1) DEFAULT NULL,
  `NI3` tinyint(1) DEFAULT NULL,
  `NI4` tinyint(1) DEFAULT NULL,
  `NI5` tinyint(1) DEFAULT NULL,
  `NI6` tinyint(1) DEFAULT NULL,
  `TI1` tinyint(1) DEFAULT NULL,
  `TI2` tinyint(1) DEFAULT NULL,
  `TI3` tinyint(1) DEFAULT NULL,
  `TI4` tinyint(1) DEFAULT NULL,
  `TI5` tinyint(1) DEFAULT NULL,
  `TI6` tinyint(1) DEFAULT NULL,
  `TI7` tinyint(1) DEFAULT NULL,
  `TI8` tinyint(1) DEFAULT NULL,
  `TI9` tinyint(1) DEFAULT NULL,
  `TI10` tinyint(1) DEFAULT NULL,
  `TI11` tinyint(1) DEFAULT NULL,
  `TI12` tinyint(1) DEFAULT NULL,
  `TI13` tinyint(1) DEFAULT NULL,
  `TI14` tinyint(1) DEFAULT NULL,
  `TI15` tinyint(1) DEFAULT NULL,
  `SI1` tinyint(1) DEFAULT NULL,
  `SI2` tinyint(1) DEFAULT NULL,
  `SI3` tinyint(1) DEFAULT NULL,
  `SI4` tinyint(1) DEFAULT NULL,
  `SI5` tinyint(1) DEFAULT NULL,
  `UI1` tinyint(1) DEFAULT NULL,
  `UI2` tinyint(1) DEFAULT NULL,
  `UI3` tinyint(1) DEFAULT NULL,
  `UI4` tinyint(1) DEFAULT NULL,
  `UI5` tinyint(1) DEFAULT NULL,
  `UI6` tinyint(1) DEFAULT NULL,
  `UI7` tinyint(1) DEFAULT NULL,
  `UI8` tinyint(1) DEFAULT NULL,
  `UI9` tinyint(1) DEFAULT NULL,
  `UI10` tinyint(1) DEFAULT NULL,
  `UI11` tinyint(1) DEFAULT NULL,
  `UI12` tinyint(1) DEFAULT NULL,
  `UI13` tinyint(1) DEFAULT NULL,
  `UI14` tinyint(1) DEFAULT NULL,
  `UI15` tinyint(1) DEFAULT NULL,
  `UI16` tinyint(1) DEFAULT NULL,
  `UI17` tinyint(1) DEFAULT NULL,
  `UI18` tinyint(1) DEFAULT NULL,
  `UI19` tinyint(1) DEFAULT NULL,
  `UI20` tinyint(1) DEFAULT NULL,
  `UI21` tinyint(1) DEFAULT NULL,
  `UI22` tinyint(1) DEFAULT NULL,
  `UI23` tinyint(1) DEFAULT NULL,
  `UI24` tinyint(1) DEFAULT NULL,
  `UI25` tinyint(1) DEFAULT NULL,
  `UI26` tinyint(1) DEFAULT NULL,
  `UI27` tinyint(1) DEFAULT NULL,
  `UI28` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`ID`),
  KEY `userID` (`userID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotTransition_Prediction_Result_Analysis
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotTransition_Prediction_Result_Analysis`;

CREATE TABLE `NextSlotTransition_Prediction_Result_Analysis` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `run_id` int(11) DEFAULT NULL,
  `feature_combination` varchar(255) DEFAULT NULL,
  `final` tinyint(1) DEFAULT NULL,
  `accuracy` double DEFAULT NULL,
  `precis` double DEFAULT NULL,
  `recall` double DEFAULT NULL,
  `fscore` double DEFAULT NULL,
  `kappa_random` double DEFAULT NULL,
  `kappa_histogram` double DEFAULT NULL,
  `kappa_dominating` double DEFAULT NULL,
  `mcc` double DEFAULT NULL,
  `frequency_of_top_class` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `run_id` (`run_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



# Dump of table NextSlotTransition_Prediction_Run
# ------------------------------------------------------------

DROP TABLE IF EXISTS `NextSlotTransition_Prediction_Run`;

CREATE TABLE `NextSlotTransition_Prediction_Run` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` int(11) DEFAULT NULL,
  `start_timestamp` varchar(30) DEFAULT NULL,
  `end_timestamp` varchar(30) DEFAULT NULL,
  `selected_algorithm` varchar(30) DEFAULT NULL,
  `selected_metric` varchar(30) DEFAULT NULL,
  `number_of_optimization_data` int(11) DEFAULT NULL,
  `number_of_training_data` int(11) DEFAULT NULL,
  `number_of_test_data` int(11) DEFAULT NULL,
  `number_of_total_data` int(11) DEFAULT NULL,
  `optimization_array` longtext,
  `training_array` longtext,
  `test_array` longtext,
  `is_network` tinyint(1) DEFAULT NULL,
  `is_temporal` tinyint(1) DEFAULT NULL,
  `is_spatial` tinyint(1) DEFAULT NULL,
  `is_context` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;




/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
