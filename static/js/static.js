document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("csvFile");
  const dropZone = document.getElementById("dropZone");
  const selectedFile = document.getElementById("selectedFile");
  const continueButton = document.getElementById("continueButton");
  const uploadForm = document.getElementById("uploadForm");
  let uploadStatus = document.getElementById("uploadStatus");

  if (!fileInput || !dropZone || !selectedFile || !continueButton) {
    return;
  }

  function isCsvFile(file) {
    return Boolean(file && file.name.toLowerCase().endsWith(".csv") &&
      file.size > 0 && file.size <= 25 * 1024 * 1024);
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
      selectedFile.textContent = `Invalid file: ${file.name}. Choose a non-empty CSV no larger than 25 MB.`;
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
        if (xhr.status >= 200 && xhr.status < 300) resolve();
        else reject(new Error("The dataset upload failed. Please try again."));
      });
      xhr.addEventListener("error", function () {
        reject(new Error("The dataset upload failed. Check your connection and try again."));
      });
      xhr.send(file);
    });
  }

  uploadForm.addEventListener("submit", async function (event) {
    if (uploadForm.dataset.blobUpload !== "true") return;
    event.preventDefault();
    const file = fileInput.files[0];
    if (!isCsvFile(file)) return;
    if (!uploadStatus) {
      uploadStatus = document.createElement("p");
      uploadStatus.className = "upload-status";
      selectedFile.insertAdjacentElement("afterend", uploadStatus);
    }
    continueButton.disabled = true;
    try {
      uploadStatus.textContent = "Preparing secure upload...";
      const prepared = await requestJson(uploadForm.dataset.prepareUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, size: file.size })
      });
      const signed = await requestJson("/api/blob-upload-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(prepared)
      });
      await uploadDirectly(signed.uploadUrl, file);
      uploadStatus.textContent = "Validating dataset...";
      const completed = await requestJson(uploadForm.dataset.completeUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pathname: prepared.pathname })
      });
      window.location.assign(completed.preview_url);
    } catch (error) {
      uploadStatus.textContent = error.message;
      uploadStatus.classList.add("upload-status-error");
      continueButton.disabled = false;
    }
  });

  showSelectedFile(fileInput.files[0]);
});
