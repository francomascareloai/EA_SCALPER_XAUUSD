//+------------------------------------------------------------------+
//|                                                  StringUtils.mqh |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"

// Check if string contains substring
bool StringContains(string text, string substring) {
    return StringFind(text, substring) != -1;
}

// Replace substring in string
string StringReplaceAll(string text, string search, string replace) {
    int pos = 0;
    while ((pos = StringFind(text, search, pos)) != -1) {
        text = StringSubstr(text, 0, pos) + replace + StringSubstr(text, pos + StringLen(search));
        pos += StringLen(replace);
    }
    return text;
}

// Split string by delimiter
string[] StringSplitToArray(string text, string delimiter) {
    string result[];
    int count = StringSplit(text, delimiter, result);
    return result;
}

// Join array to string
string ArrayToStringJoin(string array[], string delimiter = ",") {
    string result = "";
    for (int i = 0; i < ArraySize(array); i++) {
        if (i > 0) {
            result += delimiter;
        }
        result += array[i];
    }
    return result;
}

// Trim whitespace from string
string StringTrim(string text) {
    // Remove leading whitespace
    while (StringGetCharacter(text, 0) == ' ' || StringGetCharacter(text, 0) == '\t') {
        text = StringSubstr(text, 1);
    }
    
    // Remove trailing whitespace
    while (StringGetCharacter(text, StringLen(text) - 1) == ' ' || StringGetCharacter(text, StringLen(text) - 1) == '\t') {
        text = StringSubstr(text, 0, StringLen(text) - 1);
    }
    
    return text;
}