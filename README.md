## ﻿**﻿﻿﻿﻿﻿Objective**
To indicate those claiming transactions which have a possibility to fall in the fraud bucket indicating that the claim might potentially need to be investigated. Note that this does not directly indicate that the claim is a fraud or not since the historical data does not have a direct indicator of fraud.

## Features used to detect potential fraud
The following features, derived from the data, are used in order to detect potential fraud:
- Customer Number
- Age of the customer
- Time since last claim
- Last claim value
- Average value of all previous claims
- Total Claims till date
- Rejected Claim Ratio
- Average claim confirmation time
- Customer Location
- Gender
- Policy Details
- Claim Category
- Product
- Premium
- Marriage Code
- Occupation Code
- Time till policy expiration
- Number of policies bought in the last 3 months
- Number of outstanding premium policies
- Average outstanding premium
- Days since last premium paid
- "As of" claim details
- Policy start date to registration date of the claim
- REJECT CODE

## Model Used
Deep Neural Network (Keras-Tensorflow)
 
## How to Run?
### Software Requirements
Minimum Python 3.5 with all required libraries.
Use 'pip install <library-name>' to install required libraries
Libraries required for this solution are: