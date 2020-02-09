import { Component, OnInit, ViewChild, Injectable, Inject } from '@angular/core';
import {MatPaginator, MatTableDataSource} from '@angular/material';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog, MatDialogRef, MAT_DIALOG_DATA} from '@angular/material/dialog';


const httpOptions = {
  headers: new HttpHeaders({
    'Content-Type': 'application/json' ,
    'Access-Control-Allow-Origin': 'true'
  })
};
export interface UserData {
  Label: string;
  EventId: number;
  DER_mass_MMC: number;
  DER_mass_transverse_met_lep: number;
  DER_mass_vis: number;
}
export interface DialogData {
  Rainfall: number;
}
interface SearchResults {
  prediction: number;
}


@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.css'],
})

export class AppComponent implements OnInit {

title: 'AgroB';
totalAngularPackages;
Rainfall: number;


    constructor(public dialog: MatDialog) {}

    openDialog(): void {
      // this.http.get<SearchResults>('http://aae72774.ngrok.io/predict', httpOptions).subscribe(data => {
      // this.Rainfall = data.prediction;
      const dialogRef = this.dialog.open(DialogOverviewExampleDialog, {
        width : '250px',
        data : {Rainfall: 5112162}
    });
// });
}



  ngOnInit() {
 }
}

@Component({
  templateUrl: 'dialog-overview-example-dialog.html'
})

export class DialogOverviewExampleDialog {
  constructor(
    public dialogRef: MatDialogRef<DialogOverviewExampleDialog>,
    @Inject(MAT_DIALOG_DATA) public data: DialogData) {}
    onNoClick(): void {
      this.dialogRef.close();
    }
}



