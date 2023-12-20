//
// Created by qzz on 2023/12/20.
//

#ifndef BRIDGE_LEARNING_PLAYCC_FILE_H_
#define BRIDGE_LEARNING_PLAYCC_FILE_H_

#include <cstdint>
#include <string>
#include <memory>

#include "absl/strings/string_view.h"

namespace file{
// A simple file abstraction. Needed for compatibility with Google's libraries.
class File {
 public:
  File(const std::string& filename, const std::string& mode);

  // File is move only.
  File(File&& other);
  File& operator=(File&& other);
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  ~File();  // Flush and Close.

  bool Flush();  // Flush the buffer to disk.

  std::int64_t Tell();  // Offset of the current point in the file.
  bool Seek(std::int64_t offset);  // Move the current point.

  std::string Read(std::int64_t count);  // Read count bytes.
  std::string ReadContents();  // Read the entire file.

  bool Write(absl::string_view str);  // Write to the file.

  std::int64_t Length();  // Length of the entire file.

 private:
  bool Close();  // Close the file. Use the destructor instead.

  class FileImpl;
  std::unique_ptr<FileImpl> fd_;
};

// Reads the file at filename to a string. Dies if this doesn't succeed.
std::string ReadContentsFromFile(const std::string& filename,
                                 const std::string& mode);

// Write the string contents to the file. Dies if it doesn't succeed.
void WriteContentsToFile(const std::string& filename, const std::string& mode,
                         const std::string& contents);

bool Exists(const std::string& path);  // Does the file/directory exist?
bool IsDirectory(const std::string& path);  // Is it a directory?
bool Mkdir(const std::string& path, int mode = 0755);  // Make a directory.
bool Mkdirs(const std::string& path, int mode = 0755);  // Mkdir recursively.
bool Remove(const std::string& path);  // Remove/delete the file/directory.

std::string GetEnv(const std::string& key, const std::string& default_value);
std::string GetTmpDir();

}
#endif //BRIDGE_LEARNING_PLAYCC_FILE_H_
