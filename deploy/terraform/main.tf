provider "google" {
  project     = "vertex-ai-um"
  region      = "asia-southeast1"
  credentials = "./vertex-ai-um-5d040717d412.json"
}

resource "google_compute_address" "baf_address" {
  name = "baf-address"
}

resource "google_compute_instance" "baf" {
  name         = "baf"
  machine_type = "e2-small"
  zone         = "asia-southeast1-a"

  boot_disk {
    initialize_params {
      size  = 75
      type  = "pd-balanced"
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = "default"

    access_config {
      nat_ip = google_compute_address.baf_address.address
    }
  }

  metadata = {
    ssh-keys = "ubuntu:${file("./gce.pub")}"
  }

  tags = ["mlflow-server", "streamlit-server"]
}

resource "google_compute_firewall" "baf-firewall" {
  name    = "baf-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["5002", "8502"]
  }

  source_ranges = ["0.0.0.0/0"]

  target_tags   = ["mlflow-server", "streamlit-server"]
}
