import { Component, OnInit, ViewChild, Injectable } from '@angular/core';
import {MatPaginator, MatTableDataSource} from '@angular/material';
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

  ngOnInit() {

 }
}

