//+------------------------------------------------------------------+
//|                                                    FileUtils.mqh |
//|                        EA_SCALPER_XAUUSD Library                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "EA_SCALPER_XAUUSD"
#property link      ""
#property version   "1.00"

// Write data to file
bool WriteToFile(string filename, string data, int fileType = FILE_WRITE) {
    int handle = FileOpen(filename, fileType);
    if (handle != INVALID_HANDLE) {
        FileWriteString(handle, data);
        FileClose(handle);
        return true;
    }
    return false;
}

// Read data from file
string ReadFromFile(string filename) {
    string result = "";
    int handle = FileOpen(filename, FILE_READ);
    if (handle != INVALID_HANDLE) {
        result = FileReadString(handle, FileSize(handle));
        FileClose(handle);
    }
    return result;
}

// Check if file exists
bool FileExists(string filename) {
    int handle = FileOpen(filename, FILE_READ);
    if (handle != INVALID_HANDLE) {
        FileClose(handle);
        return true;
    }
    return false;
}

// Get file size
long GetFileSize(string filename) {
    long size = 0;
    int handle = FileOpen(filename, FILE_READ);
    if (handle != INVALID_HANDLE) {
        size = FileSize(handle);
        FileClose(handle);
    }
    return size;
}

// Delete file
bool DeleteFile(string filename) {
    return FileDelete(filename);
}

// List files in directory
string[] ListFilesInDirectory(string directory) {
    string files[];
    int count = 0;
    
    long handle = FileOpen(directory, FILE_COMMON);
    if (handle != INVALID_HANDLE) {
        while (!FileIsEnding(handle)) {
            string filename = FileReadString(handle);
            if (filename != "") {
                ArrayResize(files, count + 1);
                files[count] = filename;
                count++;
            }
        }
        FileClose(handle);
    }
    
    return files;
}