import { Component, OnInit, ViewChild, Injectable } from '@angular/core';
import {MatPaginator, MatTableDataSource} from '@angular/material';
import { HttpClient } from '@angular/common/http';

export interface UserData {
  Label: string;
  EventId: number;
  DER_mass_MMC: number;
  DER_mass_transverse_met_lep: number;
  DER_mass_vis: number;
}
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  title = 'higgs-boson';
  totalAngularPackages;

    constructor(private http: HttpClient) { }

  ngOnInit() {
    this.http.get<SearchResults>('https://api.npms.io/v2/search?q=scope:angular').subscribe(data => {
    this.totalAngularPackages = data.total;
})
 }
}

interface SearchResults {
  total: number;
  results: Array<object>;
}