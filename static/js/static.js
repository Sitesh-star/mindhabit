document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("csvFile");
  const dropZone = document.getElementById("dropZone");
  const selectedFile = document.getElementById("selectedFile");
  const continueButton = document.getElementById("continueButton");
  const uploadForm = document.getElementById("uploadForm");
  let uploadStatus = document.getElementById("uploadStatus");
  let directUploadRunning = false;

  if (!fileInput || !dropZone || !selectedFile || !continueButton) {
    return;
  }

  function fileValidationError(file) {
    if (!file) return null;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      return "Invalid file format. Please choose a CSV file.";
    }
    if (file.size <= 0) return "The selected CSV file is empty.";
    if (file.size > 25 * 1024 * 1024) {
      return "File is too large. Please choose a CSV file smaller than 25 MB.";
    }
    return null;
  }

  function isCsvFile(file) {
    return Boolean(file && !fileValidationError(file));
  }

  function showSelectedFile(file) {
    const valid = isCsvFile(file);
    dropZone.classList.toggle("has-file", valid);
    dropZone.classList.toggle("has-error", Boolean(file) && !valid);
    continueButton.disabled = !valid;

    if (!file) {
      selectedFile.textContent = "No file selected";
    } else if (valid) {
      selectedFile.textContent = `Selected file: ${file.name}`;
    } else {
      selectedFile.textContent = fileValidationError(file);
    }
  }

  fileInput.addEventListener("change", function () {
    showSelectedFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      dropZone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      dropZone.classList.remove("drag-active");
    });
  });

  dropZone.addEventListener("drop", function (event) {
    const file = event.dataTransfer.files[0];
    if (!file) {
      showSelectedFile(null);
      return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    showSelectedFile(file);
  });

  dropZone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  function requestJson(url, options) {
    return fetch(url, options).then(async function (response) {
      const payload = await response.json().catch(function () { return {}; });
      if (!response.ok) {
        throw new Error(payload.error || "The upload could not be completed.");
      }
      return payload;
    });
  }

  function uploadDirectly(url, file) {
    return new Promise(function (resolve, reject) {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", url);
      xhr.setRequestHeader("Content-Type", "text/csv");
      xhr.upload.addEventListener("progress", function (event) {
        if (event.lengthComputable && uploadStatus) {
          uploadStatus.textContent = `Uploading dataset... ${Math.round(event.loaded / event.total * 100)}%`;
        }
      });
      xhr.addEventListener("load", function () {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const payload = JSON.parse(xhr.responseText);
            const blob = payload.blob || payload;
            if (!blob.pathname || !blob.url) {
              throw new Error("Missing canonical Blob metadata.");
            }
            resolve(blob);
          } catch (error) {
            reject(new Error("Blob uploaded, but its storage reference was unreadable."));
          }
        } else reject(new Error("The dataset upload failed. Please try again."));
      });
      xhr.addEventListener("error", function () {
        reject(new Error("The dataset upload failed. Check your connection and try again."));
      });
      xhr.send(file);
    });
  }

  async function startDirectUpload() {
    if (uploadForm.dataset.blobUpload !== "true" || directUploadRunning) return;
    const file = fileInput.files[0];
    if (!isCsvFile(file)) return;
    if (!uploadStatus) {
      uploadStatus = document.createElement("p");
      uploadStatus.className = "upload-status";
      selectedFile.insertAdjacentElement("afterend", uploadStatus);
    }
    directUploadRunning = true;
    continueButton.disabled = true;
    try {
      uploadStatus.textContent = "Preparing secure upload...";
      const prepared = await requestJson(uploadForm.dataset.prepareUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: file.name,
          size: file.size,
          content_type: file.type || "text/csv"
        })
      });
      const signed = await requestJson("/api/blob-upload-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(prepared)
      });
      const uploadedBlob = await uploadDirectly(signed.uploadUrl, file);
      uploadStatus.textContent = "Validating dataset...";
      const completed = await requestJson(uploadForm.dataset.completeUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          storage: "blob",
          requested_pathname: prepared.pathname,
          pathname: uploadedBlob.pathname,
          url: uploadedBlob.url,
          filename: prepared.filename
        })
      });
      window.location.assign(completed.preview_url);
    } catch (error) {
      uploadStatus.textContent = error.message;
      uploadStatus.classList.add("upload-status-error");
      directUploadRunning = false;
      continueButton.disabled = false;
    }
  }

  if (uploadForm.dataset.blobUpload === "true") {
    uploadForm.addEventListener("submit", function (event) {
      event.preventDefault();
      startDirectUpload();
    });
    continueButton.addEventListener("click", startDirectUpload);
  }

  showSelectedFile(fileInput.files[0]);
});
